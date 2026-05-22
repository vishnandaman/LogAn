import os
import json
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP, Context

from logan.mcp.stdout_guard import suppress_stdout

logger = logging.getLogger("logan.mcp")

def _default_output_dir() -> str:
    """Return a timestamped output directory under ~/.logan/runs/."""
    base = os.path.join(Path.home(), ".logan", "runs")
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base, run_name)


# ---------------------------------------------------------------------------
# Server state: holds the cached ML model across tool calls
# ---------------------------------------------------------------------------

class _ServerState:
    def __init__(self):
        self._model_manager = None
        self._model_type = None
        self._model_name = None

    def get_or_create_model_manager(self, model_type, model):
        """Return cached ModelManager, creating one on first call."""
        from logan.log_diagnosis.models import ModelManager

        type_val = model_type.value if hasattr(model_type, "value") else str(model_type)
        model_val = model.value if hasattr(model, "value") else str(model)

        if (
            self._model_manager is not None
            and self._model_type == type_val
            and self._model_name == model_val
        ):
            return self._model_manager

        with suppress_stdout():
            mgr = ModelManager(model_type, model)
        self._model_manager = mgr
        self._model_type = type_val
        self._model_name = model_val
        return mgr


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(server: FastMCP):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Pandarallel uses multiprocessing.fork() from within executor threads,
    # which is unsafe in a multi-threaded asyncio process and causes deadlocks.
    os.environ["LOGAN_DISABLE_PANDARALLEL"] = "1"
    state = _ServerState()
    try:
        yield state
    finally:
        state._model_manager = None


# ---------------------------------------------------------------------------
# FastMCP instance
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "logan",
    instructions=(
        "LogAn is an intelligent log analysis tool. Use analyze_logs to run "
        "the full pipeline on log files and get golden signal classifications. "
        "Use extract_templates to quickly find unique log patterns via Drain3 "
        "without ML classification."
    ),
    lifespan=_lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_model_type(model_type_str: str):
    from logan.log_diagnosis.models import ModelType
    try:
        return ModelType(model_type_str)
    except ValueError:
        raise ValueError(
            f"Invalid model_type: '{model_type_str}'. "
            f"Choose from: {', '.join(m.value for m in ModelType)}"
        )


def _resolve_model(model_str: str):
    from logan.log_diagnosis.models.model_zero_shot_classifer import ZeroShotModels
    for m in ZeroShotModels:
        if m.name.lower() == model_str.lower() or m.value == model_str:
            return m
    return model_str


def _read_json_file(path: str):
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Tool: analyze_logs
# ---------------------------------------------------------------------------

@mcp.tool()
async def analyze_logs(
    files: list[str],
    output_dir: str = "",
    time_range: str = "all-data",
    model_type: str = "zero_shot",
    model: str = "crossencoder",
    debug_mode: bool = True,
    process_all_files: bool = False,
    process_log_files: bool = True,
    process_txt_files: bool = False,
    clean_up: bool = False,
    ctx: Context = None,
) -> dict:
    """Run the full LogAn analysis pipeline on log files.

    Performs preprocessing, Drain3 templatization, and ML-based anomaly
    detection (golden signals + fault categories). Returns each unique
    log template with its representative log line, golden signal, fault
    categories, and occurrence count.

    Model loading on the first call may take 30-60 seconds; subsequent
    calls reuse the cached model.

    Args:
        files: Paths to log files or directories to analyze.
        output_dir: Directory where reports and artifacts are written. Defaults to ~/.logan/runs/<timestamp>.
        time_range: Time range filter. Options: all-data, 1-day, 2-day, ..., 1-week, 2-week, 1-month.
        model_type: Model type for classification. Options: zero_shot, similarity, custom.
        model: Model name. Built-in: crossencoder, bart. Or a HuggingFace model name.
        debug_mode: Save additional debug artifacts.
        process_all_files: Process all text-based files regardless of extension.
        process_log_files: Process .log files found in directories.
        process_txt_files: Process .txt files found in directories.
        clean_up: Remove existing output_dir before running.
    """
    loop = asyncio.get_event_loop()
    state: _ServerState = ctx.request_context.lifespan_context

    if not output_dir:
        output_dir = _default_output_dir()

    for f in files:
        if not os.path.exists(f):
            return {"status": "error", "message": f"File not found: {f}"}

    resolved_model_type = _resolve_model_type(model_type)
    resolved_model = _resolve_model(model)
    debug_str = "true" if debug_mode else "false"

    from logan.log_diagnosis.utils import prepare_output_dir
    prepare_output_dir(output_dir, clean_up)

    # Step 1: preprocessing
    await ctx.report_progress(0, 3)
    await ctx.info("Preprocessing log files...")

    def _run_preprocessing():
        from logan.preprocessing.preprocessing import Preprocessing
        with suppress_stdout():
            pp = Preprocessing(debug_str)
            pp.preprocess(
                files, time_range, output_dir,
                process_all_files, process_log_files, process_txt_files,
            )
        return pp.df

    df = await loop.run_in_executor(None, _run_preprocessing)

    if df is None or len(df) == 0:
        return {
            "status": "error",
            "message": "No log lines could be extracted from the input files.",
        }

    total_log_lines = len(df)

    # Step 2: Drain3 templatization
    await ctx.report_progress(1, 3)
    await ctx.info("Generating log templates (Drain3)...")

    def _run_drain():
        from logan.drain.run_drain import Templatizer
        with suppress_stdout():
            drain_config = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "drain", "drain3.ini"
            )
            templatizer = Templatizer(debug_mode=debug_str, config_path=drain_config)
            templatizer.miner(df, output_dir)
        return templatizer.df

    templatized_df = await loop.run_in_executor(None, _run_drain)
    templates_found = templatized_df["test_ids"].nunique() if "test_ids" in templatized_df.columns else 0

    # Step 3: anomaly detection
    await ctx.report_progress(2, 3)
    await ctx.info("Detecting anomalies (model loading may take 30-60s on first call)...")

    def _run_anomaly():
        from logan.log_diagnosis.anomaly import Anomaly

        model_mgr = state.get_or_create_model_manager(resolved_model_type, resolved_model)

        anomaly = object.__new__(Anomaly)
        anomaly.debug_mode = debug_str
        anomaly.model_manager = model_mgr

        with suppress_stdout():
            anomaly.get_anomaly_report(templatized_df, output_dir)

    await loop.run_in_executor(None, _run_anomaly)

    await ctx.report_progress(3, 3)

    # Build structured results from the signal map written by the pipeline
    signal_map = _read_json_file(
        os.path.join(output_dir, "developer_debug_files", "temp_id_to_signal_map.json")
    )

    # Build per-template results using the summary: representative log line,
    # golden signal, fault categories, occurrence count.
    # The summary DataFrame was rendered into the HTML report; we reconstruct
    # from the templatized_df which is still in memory.
    gs_distribution = {}
    results = []

    if signal_map:
        # signal_map keys are stringified tuples: "('tid', 'file_name')"
        # Build occurrence counts from the templatized DataFrame
        counts = templatized_df.groupby(["test_ids", "file_names"]).agg(
            log_line=("text", "first"),
            occurrences=("text", "size"),
        ).reset_index()

        for _, row in counts.iterrows():
            tid = str(row["test_ids"])
            fname = row["file_names"]
            key = str((tid, fname))
            gs, fault = signal_map.get(key, ["Info", ["other"]])

            gs_distribution[gs] = gs_distribution.get(gs, 0) + int(row["occurrences"])

            log_line = row["log_line"]
            if isinstance(log_line, str):
                log_line = log_line.replace("&#13;&#10;", "\n").strip()

            results.append({
                "template_id": tid,
                "log_line": log_line,
                "golden_signal": gs,
                "fault_categories": fault[0] if isinstance(fault, list) and fault else fault,
                "occurrences": int(row["occurrences"]),
                "file_name": fname,
            })

        results.sort(key=lambda r: r["occurrences"], reverse=True)

    return {
        "status": "success",
        "total_log_lines": total_log_lines,
        "templates_found": templates_found,
        "golden_signal_distribution": gs_distribution,
        "output_dir": output_dir,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Tool: extract_templates
# ---------------------------------------------------------------------------

@mcp.tool()
async def extract_templates(
    files: list[str],
    output_dir: str = "",
    time_range: str = "all-data",
    process_all_files: bool = False,
    process_log_files: bool = True,
    process_txt_files: bool = False,
    clean_up: bool = False,
    ctx: Context = None,
) -> dict:
    """Extract unique log patterns from log files using Drain3 (no ML classification).

    Runs preprocessing and Drain3 templatization to identify recurring log
    patterns. Fast (typically under a few seconds) — use this to quickly
    understand log structure before deciding whether to run full analysis.

    Args:
        files: Paths to log files or directories to process.
        output_dir: Directory where artifacts are written. Defaults to ~/.logan/runs/<timestamp>. Override the base path with the LOGAN_OUTPUT_DIR environment variable.
        time_range: Time range filter. Options: all-data, 1-day, 2-day, ..., 1-week, 2-week, 1-month.
        process_all_files: Process all text-based files regardless of extension.
        process_log_files: Process .log files found in directories.
        process_txt_files: Process .txt files found in directories.
        clean_up: Remove existing output_dir before running.
    """
    loop = asyncio.get_event_loop()

    if not output_dir:
        output_dir = _default_output_dir()

    for f in files:
        if not os.path.exists(f):
            return {"status": "error", "message": f"File not found: {f}"}

    from logan.log_diagnosis.utils import prepare_output_dir
    prepare_output_dir(output_dir, clean_up)

    # Step 1: preprocessing
    await ctx.report_progress(0, 2)
    await ctx.info("Preprocessing log files...")

    def _run_preprocessing():
        from logan.preprocessing.preprocessing import Preprocessing
        with suppress_stdout():
            pp = Preprocessing("true")
            pp.preprocess(
                files, time_range, output_dir,
                process_all_files, process_log_files, process_txt_files,
            )
        return pp.df

    df = await loop.run_in_executor(None, _run_preprocessing)

    if df is None or len(df) == 0:
        return {
            "status": "error",
            "message": "No log lines could be extracted from the input files.",
        }

    total_log_lines = len(df)

    # Step 2: Drain3 templatization
    await ctx.report_progress(1, 2)
    await ctx.info("Extracting log templates (Drain3)...")

    def _run_drain():
        from logan.drain.run_drain import Templatizer
        from drain3.template_miner import TemplateMiner
        from drain3.template_miner_config import TemplateMinerConfig

        with suppress_stdout():
            drain_config = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "drain", "drain3.ini"
            )
            templatizer = Templatizer(debug_mode="true", config_path=drain_config)
            templatizer.miner(df, output_dir)

            # Extract template patterns from Drain3's internal clusters
            config = TemplateMinerConfig()
            config.load(drain_config)
            tm = TemplateMiner(config=config)
            for log in df["truncated_log"].values:
                tm.add_log_message(log)
            clusters = tm.drain.clusters

        return templatizer.df, clusters

    templatized_df, clusters = await loop.run_in_executor(None, _run_drain)

    await ctx.report_progress(2, 2)

    templates = []
    for cluster in clusters:
        templates.append({
            "template_id": str(cluster.cluster_id),
            "template": cluster.get_template(),
            "occurrences": cluster.size,
        })

    templates.sort(key=lambda t: t["occurrences"], reverse=True)

    return {
        "status": "success",
        "total_log_lines": total_log_lines,
        "templates_found": len(templates),
        "output_dir": output_dir,
        "templates": templates,
    }

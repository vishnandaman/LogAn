import os
import json

import pyarrow as pa
import pyarrow.parquet as pq


ENTRIES_CHUNK_SIZE = 10_000


def _flatten_fault(fault):
    if isinstance(fault, list):
        inner = fault[0] if fault else None
        if isinstance(inner, list):
            return inner[0] if inner else "other"
        return inner if inner else "other"
    return str(fault) if fault else "other"


class LogStore:
    """
    Stores the structured Drain3 output so log lines don't need to be re-parsed
    on every analysis run.

    Two Parquet files are written under output_dir/store/:
      templates.parquet  — one row per unique log pattern, with golden signal and
                           fault category filled in after ML classification.
      log_entries.parquet — one row per log line: template_id + variable tokens +
                            timestamp + file source.  Full text is reconstructed
                            on demand by substituting variables into the template.

    JSON equivalents are written to developer_debug_files/ so explorer.html can
    load them without a server.
    """

    TEMPLATES_FILE = "templates.parquet"
    ENTRIES_FILE = "log_entries.parquet"
    TEMPLATES_JSON = "store_templates.json"
    ENTRIES_JSON_PREFIX = "store_entries"

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.store_dir = os.path.join(output_dir, "store")
        self.debug_dir = os.path.join(output_dir, "developer_debug_files")
        os.makedirs(self.store_dir, exist_ok=True)

        self._templates: dict[int, dict] = {}
        self._entries: list[dict] = []

    # ------------------------------------------------------------------
    # Variable extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_variables(log_line: str, template_str: str) -> list[str]:
        """
        Return the variable tokens from log_line given its Drain3 template.

        Drain3 tokenises by whitespace and replaces variable positions with <*>.
        This method does a linear scan: static tokens are skipped, wildcard
        positions are collected.  The last wildcard absorbs any trailing extra
        tokens so that a <*> at the end of the template always captures the
        full tail of the log line.
        """
        log_tokens = log_line.split()
        tmpl_tokens = template_str.split()
        variables: list[str] = []

        li = 0
        for ti, tmpl_tok in enumerate(tmpl_tokens):
            if li >= len(log_tokens):
                break
            if tmpl_tok == "<*>":
                is_final_token = (ti == len(tmpl_tokens) - 1)
                if is_final_token and li < len(log_tokens):
                    # Last template token is a wildcard — absorb all remaining log tokens.
                    variables.append(" ".join(log_tokens[li:]))
                    li = len(log_tokens)
                else:
                    variables.append(log_tokens[li])
                    li += 1
            else:
                li += 1

        return variables

    # ------------------------------------------------------------------
    # Building the store
    # ------------------------------------------------------------------

    def build_from_df(self, df, temp_id_to_signal_map: dict):
        """
        Populate templates and log_entries from the processed DataFrame.

        df must have columns: test_ids, template_str, variables (JSON string),
        epoch, file_names.  golden_signal must also be present (added by
        process_data before this is called).

        temp_id_to_signal_map: (template_id, file_name) -> (golden_signal, fault_list)
        """
        required = {"test_ids", "template_str", "variables", "epoch", "file_names"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing columns needed by LogStore: {missing}")

        # --- templates ---
        grp = (
            df.groupby("test_ids")
            .agg(template=("template_str", "first"), occurrence_count=("test_ids", "count"))
            .reset_index()
        )
        for _, row in grp.iterrows():
            try:
                tid = int(row["test_ids"])
            except (ValueError, TypeError):
                continue
            tmpl = str(row["template"]) if row["template"] else ""
            self._templates[tid] = {
                "template_id": tid,
                "template": tmpl,
                "var_count": tmpl.count("<*>"),
                "golden_signal": None,
                "fault_category": None,
                "occurrence_count": int(row["occurrence_count"]),
            }

        self._update_signals(temp_id_to_signal_map)

        # --- log entries (vectorised) ---
        subset = df[["test_ids", "epoch", "variables", "file_names"]].copy()
        subset = subset.rename(
            columns={
                "test_ids": "template_id",
                "epoch": "timestamp",
                "file_names": "file_source",
            }
        )
        # Coerce types
        subset["template_id"] = subset["template_id"].apply(
            lambda v: int(v) if str(v).lstrip("-").isdigit() else -1
        )
        subset["timestamp"] = subset["timestamp"].fillna(0).astype("int64")
        subset["variables"] = subset["variables"].fillna("[]").astype(str)
        subset["file_source"] = subset["file_source"].fillna("").astype(str)

        self._entries = subset.to_dict("records")

    def _update_signals(self, temp_id_to_signal_map: dict):
        signal_votes: dict[int, dict] = {}
        fault_by_tid: dict[int, str] = {}

        for (tid_raw, _), (gs, fault) in temp_id_to_signal_map.items():
            try:
                tid = int(tid_raw)
            except (ValueError, TypeError):
                continue
            votes = signal_votes.setdefault(tid, {})
            votes[gs] = votes.get(gs, 0) + 1
            if tid not in fault_by_tid:
                fault_by_tid[tid] = _flatten_fault(fault)

        for tid, votes in signal_votes.items():
            if tid in self._templates:
                self._templates[tid]["golden_signal"] = max(votes, key=votes.get)
                self._templates[tid]["fault_category"] = fault_by_tid.get(tid, "other")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_parquet(self):
        """Write templates.parquet and log_entries.parquet to store_dir."""
        if self._templates:
            tbl = pa.Table.from_pylist(list(self._templates.values()))
            pq.write_table(tbl, os.path.join(self.store_dir, self.TEMPLATES_FILE))

        if self._entries:
            tbl = pa.Table.from_pylist(self._entries)
            pq.write_table(tbl, os.path.join(self.store_dir, self.ENTRIES_FILE))

    def save_json_for_explorer(self) -> dict:
        """
        Write JSON files consumed by explorer.html and return metadata that
        the Jinja2 template embeds so the page knows which files to fetch.

        Returns:
            {
                "total_templates": int,
                "total_entries":   int,
                "chunks": [{"file": str, "count": int}, ...]
            }
        """
        # templates JSON (one small file)
        templates_out = list(self._templates.values())
        with open(os.path.join(self.debug_dir, self.TEMPLATES_JSON), "w") as fh:
            json.dump(templates_out, fh)

        # entries JSON (chunked)
        chunks = []
        for chunk_idx in range(0, max(len(self._entries), 1), ENTRIES_CHUNK_SIZE):
            chunk = self._entries[chunk_idx : chunk_idx + ENTRIES_CHUNK_SIZE]
            if not chunk:
                break
            fname = f"{self.ENTRIES_JSON_PREFIX}_{len(chunks) + 1}.json"
            with open(os.path.join(self.debug_dir, fname), "w") as fh:
                json.dump(chunk, fh)
            chunks.append(
                {
                    "file": f"../developer_debug_files/{fname}",
                    "count": len(chunk),
                }
            )

        return {
            "total_templates": len(templates_out),
            "total_entries": len(self._entries),
            "templates_file": "../developer_debug_files/" + self.TEMPLATES_JSON,
            "chunks": chunks,
        }

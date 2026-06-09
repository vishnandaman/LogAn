from logan.idm_component_tagger.bracket_extractor import extract_bracket_tokens
import re


def match_tag(tokens, log_line,tags,default_tag):
    """Match tokens against seed keywords using substring similarity."""
    best_tag = default_tag
    best_score = 0
    for tag in tags:
        score = 0
        for tok in tokens:
            for kw in tag.get("keywords",[]):
                if kw in tok:
                    score += 1
                    break
        for pat in tag.get("patterns",[]):
            try:
                if re.search(pat, log_line,re.IGNORECASE):
                    score += 1
                    break
            except re.error:
                continue
        if score > best_score:
            best_score = score
            best_tag = tag["name"]
    return best_tag

class ComponentTagger:
    def __init__(self, config):
        self.tags = config.get("tags", [])
        self.default_tag = config.get("default_tag", "Other")

    def tag(self, df):
        """
        Tag each log line with a component using Drain3 template clusters.

        For each unique Drain3 template (test_ids):
          1. Extract bracket tokens from the representative log line
          2. Match tokens against seed keywords using substring similarity
          3. Broadcast the component label to all log lines in that template
        """
        template_tags = {}
        for tid, group in df.groupby("test_ids"):
            rep_log = str(group["text"].iloc[0])
            tokens = extract_bracket_tokens(rep_log)
            template_tags[tid] = match_tag(tokens,rep_log,self.tags,self.default_tag)

        df["component"] = df["test_ids"].map(template_tags)
        return df

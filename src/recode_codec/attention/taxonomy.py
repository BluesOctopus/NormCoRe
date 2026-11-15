def classify_statement(kind: str) -> str:
    # Map LibCST kinds to high-level categories (lowercase_underscore)
    k = kind.lower()
    if "import" in k: return "import"
    if k.startswith("class"): return "class_def"
    if k.startswith("function") or k == "def": return "function_def"
    if "assign" in k: return "assign"
    if k.startswith("if") or k == "else": return "if_stmt"
    if "for" in k or "while" in k: return "loop"
    return "expr"

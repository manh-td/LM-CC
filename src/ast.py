from tree_sitter import Parser
from tree_sitter_languages import get_language


# Node types we consider delimiters
DELIMITER_NODE_TYPES = {
    "function_definition",
    "class_definition",
    "if_statement",
    "for_statement",
    "while_statement",
    "switch_statement",
    "try_statement",
    "block",
    "return_statement",
    "break_statement",
    "continue_statement",
    "throw_statement",
}


def is_syntactic_delimiter(token, delimiter_offsets) -> bool:
    """
    Check if a token overlaps with any syntactic delimiter.
    token: dict with keys 'line', 'start_index', 'end_index'
    delimiter_offsets: list of dicts with keys 'line', 'start_index', 'end_index'
    """
    token_line = token.get("line")
    token_start = token.get("start_index")
    token_end = token.get("end_index")

    for delim in delimiter_offsets:
        delim_line = delim["line"]
        delim_start = delim["start_index"]
        delim_end = delim["end_index"]

        # Only compare tokens on the same line
        if token_line == delim_line:
            # Check for overlap
            if token_start < delim_end and token_end > delim_start:
                return True

    return False


def build_delimiter_offsets_and_tokens(code: str, language_name: str):
    """
    Parses code and returns:
      - delimiter offsets: list of dicts {line, start_index, end_index, text}
      - token offsets: list of (token_text, start_byte, end_byte)
    """
    language = get_language(language_name)
    parser = Parser()
    parser.set_language(language)
    tree = parser.parse(bytes(code, "utf8"))
    root = tree.root_node

    delimiter_offsets = []

    # Precompute line start byte offsets for mapping byte -> line/column
    lines = code.splitlines(keepends=True)
    line_start_bytes = []
    offset = 0
    for l in lines:
        line_start_bytes.append(offset)
        offset += len(l.encode("utf8"))

    def byte_to_line_col(start_byte, end_byte):
        # Find line
        line_num = 0
        for i, line_offset in enumerate(line_start_bytes):
            if start_byte >= line_offset:
                line_num = i
            else:
                break
        start_index = start_byte - line_start_bytes[line_num]
        end_index = end_byte - line_start_bytes[line_num]
        return line_num + 1, start_index, end_index

    def visit(node):
        if node.type in DELIMITER_NODE_TYPES:
            line, start_index, end_index = byte_to_line_col(node.start_byte, node.end_byte)
            delimiter_offsets.append({
                "line": line,
                "start_index": start_index,
                "end_index": end_index,
                "text": code[node.start_byte:node.end_byte]
            })

        for child in node.children:
            visit(child)

    visit(root)
    return delimiter_offsets

if __name__ == "__main__":
    # Test the function with example code
    test_code = """
    x = 0

    def hello_world():
        print("Hello, World!")

    class MyClass:
        def method(self):
            if True:
                return 42
    """

    offsets = build_delimiter_offsets_and_tokens(test_code, "python")
    print("Delimiter offsets:", offsets)
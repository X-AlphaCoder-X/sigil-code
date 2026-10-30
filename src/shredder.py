from .debug_log import log_ignored_error as _log_ignored  # handled errors still reach the log file

# Tree-sitter is optional (pip install sigilcode[ast]). Without it every file
# falls back to a head/tail outline.
try:
    import tree_sitter_python as tspython
    import tree_sitter_cpp as tscpp
    from tree_sitter import Language, Parser

    PY_LANGUAGE = Language(tspython.language())
    CPP_LANGUAGE = Language(tscpp.language())
except Exception as _ignored:  # ImportError or ABI mismatch
    _log_ignored(_ignored)
    PY_LANGUAGE = CPP_LANGUAGE = None

def get_file_outline(content: str | bytes, extension: str) -> str:
    """Extracts a high-level outline (classes/functions) from file content using Tree-Sitter."""
    if not content:
        return "# (Empty file)"

    # Handle both string and bytes input, and defensively coerce unexpected types
    if isinstance(content, bytes):
        byte_content = content
        content = content.decode("utf8", errors="replace")
    elif isinstance(content, str):
        byte_content = content.encode("utf8")
    else:
        # Failsafe for unexpected types in session context
        content = str(content)
        byte_content = content.encode("utf8")

    # tree-sitter positions are byte offsets, so the outline slices the UTF-8 bytes.
    if extension == '.py' and PY_LANGUAGE is not None:
        parser = Parser(PY_LANGUAGE)
        tree = parser.parse(byte_content)
        return _extract_python_outline(tree.root_node, _ByteText(byte_content))
    elif extension in ('.cpp', '.hpp', '.c', '.h', '.cc', '.cxx') and CPP_LANGUAGE is not None:
        parser = Parser(CPP_LANGUAGE)
        tree = parser.parse(byte_content)
        return _extract_cpp_outline(tree.root_node, _ByteText(byte_content))
    else:
        # Fallback for other languages
        lines = content.splitlines()
        if len(lines) <= 20:
            return content
        return '\n'.join(lines[:10] + ['...'] + lines[-10:])

class _ByteText:
    """content[start:end] with tree-sitter byte offsets, decoded as UTF-8.

    Why: the byte offsets were used on the decoded str, so any non-ASCII character before a
    function shifted every outline after it (`def hello(a, b)` came out as `def  b):`)."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def __getitem__(self, item: slice) -> str:
        return self._data[item].decode('utf-8', errors='replace')


def _extract_python_outline(root_node, content) -> str:
    """Extracts classes and functions from Python AST."""
    outline = []
    
    # Simple recursive traversal for signatures
    def traverse(node, depth=0):
        if node.type in ('class_definition', 'function_definition'):
            # Find the name node
            name_node = node.child_by_field_name('name')
            if name_node:
                name = content[name_node.start_byte:name_node.end_byte]
                kind = 'class' if node.type == 'class_definition' else 'def'
                
                # For functions, we might want parameters
                params = ""
                if node.type == 'function_definition':
                    param_node = node.child_by_field_name('parameters')
                    if param_node:
                        params = content[param_node.start_byte:param_node.end_byte]
                
                outline.append('  ' * depth + f"{kind} {name}{params}: ...")
                
                # Recurse into body to find nested classes/methods
                body = node.child_by_field_name('body')
                if body:
                    for child in body.children:
                        traverse(child, depth + 1)
        else:
            for child in node.children:
                traverse(child, depth)

    traverse(root_node)
    return '\n'.join(outline) if outline else "# (No Python signatures found)"

def _extract_cpp_outline(root_node, content) -> str:
    """Extracts classes, structs, and functions from C++ AST."""
    outline = []
    
    # Tree-sitter-cpp nodes of interest:
    # class_specifier, struct_specifier, function_definition, declaration (for prototypes)
    
    def traverse(node, depth=0):
        node_type = node.type
        
        if node_type in ('class_specifier', 'struct_specifier'):
            name_node = node.child_by_field_name('name')
            if name_node:
                name = content[name_node.start_byte:name_node.end_byte]
                kind = 'class' if node_type == 'class_specifier' else 'struct'
                outline.append('  ' * depth + f"{kind} {name} {{ ... }};")
                
                # Check for body (field_declaration_list)
                body = node.child_by_field_name('body')
                if body:
                    for child in body.children:
                        traverse(child, depth + 1)
                        
        elif node_type == 'function_definition':
            # Find the declarator which contains the name
            decl = node.child_by_field_name('declarator')
            if decl:
                # The declarator can be complex (e.g. MyClass::myMethod)
                # We'll just grab the text for the signature part
                sig = content[decl.start_byte:decl.end_byte]
                # Try to find the type
                type_node = node.child_by_field_name('type')
                type_prefix = content[type_node.start_byte:type_node.end_byte] + " " if type_node else ""
                outline.append('  ' * depth + f"{type_prefix}{sig} {{ ... }}")
                
        elif node_type == 'namespace_definition':
            name_node = node.child_by_field_name('name')
            name = content[name_node.start_byte:name_node.end_byte] if name_node else "(anonymous)"
            outline.append('  ' * depth + f"namespace {name} {{")
            body = node.child_by_field_name('body')
            if body:
                for child in body.children:
                    traverse(child, depth + 1)
            outline.append('  ' * depth + "}")
            
        else:
            for child in node.children:
                traverse(child, depth)

    traverse(root_node)
    return '\n'.join(outline) if outline else "# (No C++ signatures found)"

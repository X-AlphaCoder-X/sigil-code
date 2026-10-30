"""
Benchmark task definitions for Sigil Code.

Each task has:
  - id:          unique identifier
  - category:    what skill is being tested
  - difficulty:  easy / medium / hard
  - instruction: what the agent is told to do
  - setup:       shell commands to prepare the workspace (run BEFORE the agent)
  - verify:      shell commands that return exit 0 on success (run AFTER the agent)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkTask:
    id: str
    category: str
    difficulty: str
    instruction: str
    setup: str
    verify: str


TASKS: tuple[BenchmarkTask, ...] = (

    # ------------------------------------------------------------------
    # 1. FILE CREATION
    # ------------------------------------------------------------------
    BenchmarkTask(
        id="file-create-basic",
        category="file-ops",
        difficulty="easy",
        instruction="Create a file called hello.txt containing exactly the text: Hello, World!",
        setup="",
        verify='[ -f hello.txt ] && grep -qx "Hello, World!" hello.txt',
    ),

    BenchmarkTask(
        id="file-create-nested",
        category="file-ops",
        difficulty="easy",
        instruction="Create the directory structure src/utils/ and inside it create a file called helpers.py containing a Python function called greet that takes a name parameter and returns the string 'Hello, <name>!'.",
        setup="",
        verify=(
            '[ -f src/utils/helpers.py ] && '
            'python3 -c "from src.utils.helpers import greet; assert greet(\'World\') == \'Hello, World!\'"'
        ),
    ),

    # ------------------------------------------------------------------
    # 2. FILE EDITING
    # ------------------------------------------------------------------
    BenchmarkTask(
        id="edit-replace-string",
        category="file-edit",
        difficulty="easy",
        instruction="In the file config.txt, replace every occurrence of 'localhost' with '0.0.0.0'.",
        setup=(
            'echo "host=localhost\nport=8080\ndb_host=localhost\nbackup=localhost:3000" > config.txt'
        ),
        verify=(
            '! grep -q "localhost" config.txt && '
            'grep -q "0.0.0.0" config.txt && '
            '[ "$(grep -c "0.0.0.0" config.txt)" = "3" ]'
        ),
    ),

    BenchmarkTask(
        id="edit-add-function",
        category="file-edit",
        difficulty="medium",
        instruction="The file math_utils.py has an add function. Add a new function called multiply that takes two arguments a and b and returns a * b. Do not change the existing add function.",
        setup=(
            'cat > math_utils.py << \'PYEOF\'\n'
            'def add(a, b):\n'
            '    return a + b\n'
            'PYEOF'
        ),
        verify=(
            'python3 -c "'
            'from math_utils import add, multiply; '
            'assert add(2, 3) == 5; '
            'assert multiply(4, 5) == 20; '
            'assert multiply(0, 10) == 0; '
            'assert multiply(-2, 3) == -6'
            '"'
        ),
    ),

    # ------------------------------------------------------------------
    # 3. BUG FIXING
    # ------------------------------------------------------------------
    BenchmarkTask(
        id="bugfix-off-by-one",
        category="bugfix",
        difficulty="medium",
        instruction="The file fibonacci.py has a function that should return the first n Fibonacci numbers as a list. For example, fibonacci(5) should return [0, 1, 1, 2, 3]. But it has a bug. Find and fix it.",
        setup=(
            'cat > fibonacci.py << \'PYEOF\'\n'
            'def fibonacci(n):\n'
            '    if n <= 0:\n'
            '        return []\n'
            '    if n == 1:\n'
            '        return [0]\n'
            '    fibs = [0, 1]\n'
            '    for i in range(2, n + 1):\n'
            '        fibs.append(fibs[i-1] + fibs[i-2])\n'
            '    return fibs\n'
            'PYEOF'
        ),
        verify=(
            'python3 -c "'
            'from fibonacci import fibonacci; '
            'assert fibonacci(0) == []; '
            'assert fibonacci(1) == [0]; '
            'assert fibonacci(2) == [0, 1]; '
            'assert fibonacci(5) == [0, 1, 1, 2, 3]; '
            'assert fibonacci(8) == [0, 1, 1, 2, 3, 5, 8, 13]'
            '"'
        ),
    ),

    BenchmarkTask(
        id="bugfix-syntax-error",
        category="bugfix",
        difficulty="easy",
        instruction="The file broken.py has syntax errors that prevent it from running. Fix all syntax errors so that running 'python3 broken.py' prints 'All tests passed'.",
        setup=(
            'cat > broken.py << \'PYEOF\'\n'
            'def calculate(x, y)\n'
            '    result = x + y\n'
            '    return result\n'
            '\n'
            'def main():\n'
            '    total = calculate(10, 20\n'
            '    if total == 30:\n'
            '        print("All tests passed")\n'
            '    else\n'
            '        print("Failed")\n'
            '\n'
            'if __name__ == "__main__":\n'
            '    main()\n'
            'PYEOF'
        ),
        verify='python3 broken.py 2>&1 | grep -qx "All tests passed"',
    ),

    BenchmarkTask(
        id="bugfix-logic-error",
        category="bugfix",
        difficulty="medium",
        instruction="The file sorter.py has a function called bubble_sort that should sort a list in ascending order, but it produces wrong results. Find the bug and fix it. Do not replace the algorithm with a different one — fix the existing bubble sort logic.",
        setup=(
            'cat > sorter.py << \'PYEOF\'\n'
            'def bubble_sort(arr):\n'
            '    n = len(arr)\n'
            '    for i in range(n):\n'
            '        for j in range(0, n - 1):\n'
            '            if arr[j] > arr[j + 1]:\n'
            '                arr[j] = arr[j + 1]\n'
            '                arr[j + 1] = arr[j]\n'
            '    return arr\n'
            'PYEOF'
        ),
        verify=(
            'python3 -c "'
            'from sorter import bubble_sort; '
            'assert bubble_sort([3,1,2]) == [1,2,3]; '
            'assert bubble_sort([5,4,3,2,1]) == [1,2,3,4,5]; '
            'assert bubble_sort([]) == []; '
            'assert bubble_sort([1]) == [1]; '
            'assert bubble_sort([2,2,1]) == [1,2,2]'
            '"'
        ),
    ),

    # ------------------------------------------------------------------
    # 4. CODE GENERATION
    # ------------------------------------------------------------------
    BenchmarkTask(
        id="codegen-csv-parser",
        category="codegen",
        difficulty="medium",
        instruction=(
            "Create a file called csv_parser.py with a function called parse_csv that takes a "
            "filename (string) and returns a list of dictionaries. The first row of the CSV is "
            "the header. Each subsequent row becomes a dict mapping header names to values. "
            "Use only the Python standard library."
        ),
        setup=(
            'cat > data.csv << \'CSVEOF\'\n'
            'name,age,city\n'
            'Alice,30,NYC\n'
            'Bob,25,LA\n'
            'CSVEOF'
        ),
        verify=(
            'python3 -c "'
            'from csv_parser import parse_csv; '
            'rows = parse_csv(\"data.csv\"); '
            'assert len(rows) == 2; '
            'assert rows[0][\"name\"] == \"Alice\"; '
            'assert rows[0][\"age\"] == \"30\"; '
            'assert rows[0][\"city\"] == \"NYC\"; '
            'assert rows[1][\"name\"] == \"Bob\"'
            '"'
        ),
    ),

    BenchmarkTask(
        id="codegen-stack",
        category="codegen",
        difficulty="medium",
        instruction=(
            "Create a file called stack.py with a class called Stack. "
            "It should support: push(item), pop() which returns the item (raises IndexError if empty), "
            "peek() which returns the top item without removing it (raises IndexError if empty), "
            "is_empty() which returns True/False, and size() which returns the count."
        ),
        setup="",
        verify=(
            'python3 -c "'
            'from stack import Stack; '
            's = Stack(); '
            'assert s.is_empty() == True; '
            'assert s.size() == 0; '
            's.push(10); s.push(20); s.push(30); '
            'assert s.size() == 3; '
            'assert s.peek() == 30; '
            'assert s.pop() == 30; '
            'assert s.pop() == 20; '
            'assert s.size() == 1; '
            'assert s.is_empty() == False; '
            's.pop(); '
            'try:\n'
            '    s.pop()\n'
            '    assert False\n'
            'except IndexError:\n'
            '    pass; '
            'try:\n'
            '    s.peek()\n'
            '    assert False\n'
            'except IndexError:\n'
            '    pass'
            '"'
        ),
    ),

    BenchmarkTask(
        id="codegen-rest-api",
        category="codegen",
        difficulty="hard",
        instruction=(
            "Create a file called todo_api.py that implements a minimal TODO API using only "
            "the Python standard library (http.server). It should handle:\n"
            "  GET  /todos       -> return JSON list of all todos\n"
            "  POST /todos       -> create a todo from JSON body {\"title\": \"...\"}, return it with an auto-increment id and done=false\n"
            "  GET  /todos/<id>  -> return a single todo by id, or 404\n"
            "Todos are stored in memory (no database). Each todo has: id (int), title (str), done (bool)."
        ),
        setup="",
        verify=(
            'python3 << \'TESTEOF\'\n'
            'import subprocess, time, json, urllib.request, urllib.error, sys, signal, os\n'
            'proc = subprocess.Popen([sys.executable, "todo_api.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)\n'
            'time.sleep(2)\n'
            'try:\n'
            '    # GET empty\n'
            '    r = urllib.request.urlopen("http://127.0.0.1:8080/todos")\n'
            '    assert json.loads(r.read()) == [], "GET /todos should be empty"\n'
            '    # POST\n'
            '    data = json.dumps({"title": "Buy milk"}).encode()\n'
            '    req = urllib.request.Request("http://127.0.0.1:8080/todos", data=data, headers={"Content-Type": "application/json"}, method="POST")\n'
            '    r = urllib.request.urlopen(req)\n'
            '    todo = json.loads(r.read())\n'
            '    assert todo["id"] == 1\n'
            '    assert todo["title"] == "Buy milk"\n'
            '    assert todo["done"] == False\n'
            '    # GET by id\n'
            '    r = urllib.request.urlopen("http://127.0.0.1:8080/todos/1")\n'
            '    assert json.loads(r.read())["title"] == "Buy milk"\n'
            '    # 404\n'
            '    try:\n'
            '        urllib.request.urlopen("http://127.0.0.1:8080/todos/999")\n'
            '        assert False\n'
            '    except urllib.error.HTTPError as e:\n'
            '        assert e.code == 404\n'
            '    # GET all\n'
            '    r = urllib.request.urlopen("http://127.0.0.1:8080/todos")\n'
            '    assert len(json.loads(r.read())) == 1\n'
            '    print("ALL_TESTS_PASSED")\n'
            'finally:\n'
            '    proc.terminate()\n'
            '    proc.wait()\n'
            'TESTEOF\n'
            '[ $? -eq 0 ]'
        ),
    ),

    # ------------------------------------------------------------------
    # 5. SHELL / SYSTEM TASKS
    # ------------------------------------------------------------------
    BenchmarkTask(
        id="shell-find-largest",
        category="shell",
        difficulty="easy",
        instruction=(
            "There are several .txt files in the workspace. Find which .txt file has the most lines "
            "and write its filename (just the name, e.g. 'data3.txt') into a file called answer.txt."
        ),
        setup=(
            'for i in 1 2 3 4 5; do\n'
            '  head -c $((i * 50)) /dev/urandom | base64 | head -n $((i * 3)) > "data${i}.txt"\n'
            'done'
        ),
        verify=(
            'EXPECTED=$(wc -l data*.txt | sort -n | tail -2 | head -1 | awk \'{print $2}\') && '
            '[ -f answer.txt ] && '
            'ANSWER=$(cat answer.txt | tr -d "[:space:]") && '
            '[ "$ANSWER" = "$EXPECTED" ]'
        ),
    ),

    BenchmarkTask(
        id="shell-count-python-funcs",
        category="shell",
        difficulty="medium",
        instruction=(
            "Count the total number of Python function definitions (lines starting with 'def ') "
            "across ALL .py files in the project/ directory (recursively). Write just the number "
            "into a file called answer.txt."
        ),
        setup=(
            'mkdir -p project/sub\n'
            'cat > project/a.py << \'PY\'\n'
            'def foo():\n'
            '    pass\n'
            'def bar():\n'
            '    pass\n'
            'PY\n'
            'cat > project/b.py << \'PY\'\n'
            'def baz():\n'
            '    pass\n'
            'PY\n'
            'cat > project/sub/c.py << \'PY\'\n'
            'def one():\n'
            '    pass\n'
            'def two():\n'
            '    pass\n'
            'def three():\n'
            '    pass\n'
            'PY'
        ),
        verify=(
            '[ -f answer.txt ] && '
            'ANSWER=$(cat answer.txt | tr -d "[:space:]") && '
            '[ "$ANSWER" = "6" ]'
        ),
    ),

    # ------------------------------------------------------------------
    # 6. REFACTORING
    # ------------------------------------------------------------------
    BenchmarkTask(
        id="refactor-extract-function",
        category="refactor",
        difficulty="medium",
        instruction=(
            "The file process.py has duplicated logic for validating emails in two places. "
            "Extract the common validation logic into a single function called is_valid_email "
            "and make both register_user and update_email call it. All existing behavior must "
            "remain the same."
        ),
        setup=(
            'cat > process.py << \'PYEOF\'\n'
            'def register_user(name, email):\n'
            '    if "@" not in email or "." not in email.split("@")[-1]:\n'
            '        return {"error": "invalid email"}\n'
            '    return {"name": name, "email": email, "status": "registered"}\n'
            '\n'
            'def update_email(user, new_email):\n'
            '    if "@" not in new_email or "." not in new_email.split("@")[-1]:\n'
            '        return {"error": "invalid email"}\n'
            '    user["email"] = new_email\n'
            '    return user\n'
            'PYEOF'
        ),
        verify=(
            'python3 -c "'
            'from process import register_user, update_email, is_valid_email; '
            'assert is_valid_email(\"test@example.com\") == True; '
            'assert is_valid_email(\"bad\") == False; '
            'assert is_valid_email(\"no@dot\") == False; '
            'r = register_user(\"Alice\", \"a@b.c\"); assert r[\"status\"] == \"registered\"; '
            'r = register_user(\"Bob\", \"bad\"); assert r[\"error\"] == \"invalid email\"; '
            'u = {\"name\": \"X\", \"email\": \"old@o.com\"}; '
            'r = update_email(u, \"new@n.com\"); assert r[\"email\"] == \"new@n.com\"; '
            'r = update_email(u, \"bad\"); assert r[\"error\"] == \"invalid email\"'
            '"'
        ),
    ),

    # ------------------------------------------------------------------
    # 7. TESTING / TEST WRITING
    # ------------------------------------------------------------------
    BenchmarkTask(
        id="testgen-write-tests",
        category="testing",
        difficulty="hard",
        instruction=(
            "The file calculator.py has a Calculator class with add, subtract, multiply, and "
            "divide methods. Write a test file called test_calculator.py using unittest that "
            "has at least 8 test cases covering normal usage AND edge cases (division by zero "
            "should raise ValueError). All tests must pass when run with 'python3 -m unittest test_calculator -v'."
        ),
        setup=(
            'cat > calculator.py << \'PYEOF\'\n'
            'class Calculator:\n'
            '    def add(self, a, b):\n'
            '        return a + b\n'
            '\n'
            '    def subtract(self, a, b):\n'
            '        return a - b\n'
            '\n'
            '    def multiply(self, a, b):\n'
            '        return a * b\n'
            '\n'
            '    def divide(self, a, b):\n'
            '        if b == 0:\n'
            '            raise ValueError("Cannot divide by zero")\n'
            '        return a / b\n'
            'PYEOF'
        ),
        verify=(
            'python3 -m unittest test_calculator -v 2>&1 | tail -1 | grep -q "OK" && '
            'TESTS=$(python3 -m unittest test_calculator -v 2>&1 | grep -c "\\.\\.\\. ok") && '
            '[ "$TESTS" -ge 8 ]'
        ),
    ),

    # ------------------------------------------------------------------
    # 8. DATA PROCESSING
    # ------------------------------------------------------------------
    BenchmarkTask(
        id="data-json-transform",
        category="data",
        difficulty="medium",
        instruction=(
            "The file users.json contains an array of user objects with fields: name, age, city. "
            "Create a Python script called transform.py that reads users.json and writes a new "
            "file called summary.json containing: {\"total\": <count>, \"average_age\": <float rounded to 1 decimal>, "
            "\"cities\": [<sorted unique city list>]}."
        ),
        setup=(
            'cat > users.json << \'JSONEOF\'\n'
            '[\n'
            '  {"name": "Alice", "age": 30, "city": "NYC"},\n'
            '  {"name": "Bob", "age": 25, "city": "LA"},\n'
            '  {"name": "Carol", "age": 35, "city": "NYC"},\n'
            '  {"name": "Dave", "age": 28, "city": "Chicago"},\n'
            '  {"name": "Eve", "age": 22, "city": "LA"}\n'
            ']\n'
            'JSONEOF'
        ),
        verify=(
            'python3 transform.py && '
            'python3 -c "'
            'import json; '
            'd = json.load(open(\"summary.json\")); '
            'assert d[\"total\"] == 5; '
            'assert d[\"average_age\"] == 28.0; '
            'assert d[\"cities\"] == [\"Chicago\", \"LA\", \"NYC\"]'
            '"'
        ),
    ),

    # ------------------------------------------------------------------
    # 9. MULTI-FILE PROJECT
    # ------------------------------------------------------------------
    BenchmarkTask(
        id="project-fix-imports",
        category="project",
        difficulty="hard",
        instruction=(
            "The project has three files: app/main.py, app/models.py, and app/utils.py. "
            "The main.py tries to import from models and utils but has broken imports. "
            "Also, app/__init__.py is missing. Fix everything so that running "
            "'python3 -m app.main' prints 'App running: User(admin) validated'."
        ),
        setup=(
            'mkdir -p app\n'
            'cat > app/models.py << \'PY\'\n'
            'class User:\n'
            '    def __init__(self, name):\n'
            '        self.name = name\n'
            '    def __repr__(self):\n'
            '        return f"User({self.name})"\n'
            'PY\n'
            'cat > app/utils.py << \'PY\'\n'
            'def validate(user):\n'
            '    return user.name is not None and len(user.name) > 0\n'
            'PY\n'
            'cat > app/main.py << \'PY\'\n'
            'from models import User\n'
            'from utils import validate\n'
            '\n'
            'def run():\n'
            '    u = User("admin")\n'
            '    v = validate(u)\n'
            '    status = "validated" if v else "invalid"\n'
            '    print(f"App running: {u} {status}")\n'
            '\n'
            'if __name__ == "__main__":\n'
            '    run()\n'
            'PY'
        ),
        verify='python3 -m app.main 2>&1 | grep -qx "App running: User(admin) validated"',
    ),

    # ------------------------------------------------------------------
    # 10. ALGORITHM IMPLEMENTATION
    # ------------------------------------------------------------------
    BenchmarkTask(
        id="algo-binary-search",
        category="algorithm",
        difficulty="medium",
        instruction=(
            "Create a file called search.py with a function called binary_search that takes "
            "a sorted list and a target value. It should return the index of the target if found, "
            "or -1 if not found. Implement it using actual binary search (not list.index or linear scan)."
        ),
        setup="",
        verify=(
            'python3 -c "'
            'from search import binary_search; '
            'assert binary_search([1,2,3,4,5], 3) == 2; '
            'assert binary_search([1,2,3,4,5], 1) == 0; '
            'assert binary_search([1,2,3,4,5], 5) == 4; '
            'assert binary_search([1,2,3,4,5], 6) == -1; '
            'assert binary_search([], 1) == -1; '
            'assert binary_search([10], 10) == 0; '
            'assert binary_search([10], 5) == -1; '
            'assert binary_search(list(range(1000)), 500) == 500'
            '"'
        ),
    ),

    # ------------------------------------------------------------------
    # 11. DEBUGGING WITH READING
    # ------------------------------------------------------------------
    BenchmarkTask(
        id="debug-read-and-fix",
        category="debug",
        difficulty="hard",
        instruction=(
            "The file server_config.py has a function called parse_config that reads a .ini "
            "style config file and returns a dictionary. But it crashes on the provided "
            "settings.ini file. Read both files, find the bug, and fix parse_config so it "
            "works correctly. Do not modify settings.ini."
        ),
        setup=(
            'cat > settings.ini << \'INI\'\n'
            '[database]\n'
            'host = localhost\n'
            'port = 5432\n'
            '\n'
            '# This is a comment\n'
            '[server]\n'
            'debug = true\n'
            'workers = 4\n'
            '\n'
            '[logging]\n'
            'level = info\n'
            'INI\n'
            'cat > server_config.py << \'PYEOF\'\n'
            'def parse_config(filename):\n'
            '    result = {}\n'
            '    current_section = None\n'
            '    with open(filename) as f:\n'
            '        for line in f:\n'
            '            line = line.strip()\n'
            '            if line.startswith("["):\n'
            '                current_section = line[1:-1]\n'
            '                result[current_section] = {}\n'
            '            elif "=" in line:\n'
            '                key, value = line.split("=")\n'
            '                result[current_section][key.strip()] = value.strip()\n'
            'PYEOF'
        ),
        verify=(
            'python3 -c "'
            'from server_config import parse_config; '
            'c = parse_config(\"settings.ini\"); '
            'assert c[\"database\"][\"host\"] == \"localhost\"; '
            'assert c[\"database\"][\"port\"] == \"5432\"; '
            'assert c[\"server\"][\"debug\"] == \"true\"; '
            'assert c[\"server\"][\"workers\"] == \"4\"; '
            'assert c[\"logging\"][\"level\"] == \"info\"'
            '"'
        ),
    ),

    # ------------------------------------------------------------------
    # 12. GREP + ANALYSIS
    # ------------------------------------------------------------------
    BenchmarkTask(
        id="analysis-find-todos",
        category="analysis",
        difficulty="easy",
        instruction=(
            "Search all .py files in the codebase/ directory recursively for lines containing "
            "'TODO'. Create a file called todos.txt where each line has the format: "
            "'<filename>:<line_number>: <the TODO text>'. Sort by filename then line number."
        ),
        setup=(
            'mkdir -p codebase/sub\n'
            'cat > codebase/alpha.py << \'PY\'\n'
            '# TODO: add logging\n'
            'def alpha():\n'
            '    pass  # TODO: implement\n'
            'PY\n'
            'cat > codebase/sub/beta.py << \'PY\'\n'
            'def beta():\n'
            '    # TODO: handle errors\n'
            '    return 42\n'
            'PY'
        ),
        verify=(
            '[ -f todos.txt ] && '
            '[ "$(wc -l < todos.txt | tr -d " ")" = "3" ] && '
            'grep -q "alpha.py" todos.txt && '
            'grep -q "beta.py" todos.txt && '
            'grep -q "TODO" todos.txt'
        ),
    ),
)


def get_task(task_id: str) -> BenchmarkTask | None:
    for t in TASKS:
        if t.id == task_id:
            return t
    return None


def list_tasks() -> list[dict[str, str]]:
    return [
        {"id": t.id, "category": t.category, "difficulty": t.difficulty}
        for t in TASKS
    ]

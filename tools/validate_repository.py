import argparse
import ast
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import nbformat


REPO_ROOT = Path(__file__).resolve().parent.parent
EXCLUDE_DIRS = {".git", "__pycache__", ".ipynb_checkpoints"}
TEXT_SUFFIXES = {".cfg", ".ini", ".ipynb", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}
EXPECTED_NOTEBOOKS = {
    Path("M1-neuron-basics/attachments/kim-01-neuron-perceptron.ipynb"),
    Path("M1-neuron-basics/attachments/kim-01-neuron-perceptron-solution.ipynb"),
    Path("M2-training/attachments/kim-02-backprop-training.ipynb"),
    Path("M2-training/attachments/kim-02-backprop-training-solution.ipynb"),
    Path("M3-dense-networks/attachments/kim-03-mlp.ipynb"),
    Path("M3-dense-networks/attachments/kim-03-mlp-solution.ipynb"),
    Path("M4-cnn/attachments/kim-04-cnn.ipynb"),
    Path("M4-cnn/attachments/kim-04-cnn-solution.ipynb"),
    Path("M5-sequences/attachments/kim-05-sequences.ipynb"),
    Path("M5-sequences/attachments/kim-05-sequences-solution.ipynb"),
    Path("M6-autoencoders/attachments/kim-06-autoencoders.ipynb"),
    Path("M6-autoencoders/attachments/kim-06-autoencoders-solution.ipynb"),
    Path("M7-optimizers/attachments/kim-07-optimizers.ipynb"),
    Path("M7-optimizers/attachments/kim-07-optimizers-solution.ipynb"),
}


def find_files(pattern, root=None):
    root = root or REPO_ROOT
    for path in root.rglob(pattern):
        if any(excluded in path.parts for excluded in EXCLUDE_DIRS):
            continue
        yield path


def check_python_syntax():
    errors = []
    for py_file in find_files("*.py"):
        try:
            ast.parse(py_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, SyntaxError) as error:
            errors.append(f"{py_file.relative_to(REPO_ROOT)}: {error}")
    return errors


def read_notebook(nb_file):
    with nb_file.open(encoding="utf-8") as source:
        return nbformat.read(source, as_version=nbformat.NO_CONVERT)


def check_notebook_format():
    errors = []
    for nb_file in find_files("*.ipynb"):
        try:
            notebook = read_notebook(nb_file)
            nbformat.validate(notebook)
        except Exception as error:
            errors.append(f"{nb_file.relative_to(REPO_ROOT)}: {error}")
    return errors


def _strip_magics(source):
    lines = source.splitlines(keepends=True) if isinstance(source, str) else source
    return [line for line in lines if not re.match(r"^\s*[!%]", line)]


def check_code_cell_syntax():
    errors = []
    for nb_file in find_files("*.ipynb"):
        try:
            notebook = read_notebook(nb_file)
        except Exception:
            continue
        relative = nb_file.relative_to(REPO_ROOT)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            code_text = "".join(_strip_magics(cell.source))
            if not code_text.strip():
                continue
            try:
                ast.parse(code_text)
            except SyntaxError as error:
                errors.append(f"{relative}: ячейка {index}: {error}")
    return errors


def check_error_outputs():
    errors = []
    for nb_file in find_files("*.ipynb"):
        try:
            notebook = read_notebook(nb_file)
        except Exception:
            continue
        relative = nb_file.relative_to(REPO_ROOT)
        for cell_index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            for output_index, output in enumerate(cell.get("outputs", [])):
                if output.output_type == "error":
                    errors.append(
                        f"{relative}: ячейка {cell_index}, output {output_index}: "
                        f"{output.get('ename', '?')}: {output.get('evalue', '?')}"
                    )
    return errors


def check_executed_solution_cells():
    stats = {}
    for nb_file in find_files("*solution*.ipynb"):
        try:
            notebook = read_notebook(nb_file)
        except Exception:
            continue
        total_code = sum(cell.cell_type == "code" for cell in notebook.cells)
        executed = sum(
            cell.cell_type == "code" and cell.execution_count is not None
            for cell in notebook.cells
        )
        stats[nb_file.relative_to(REPO_ROOT)] = {"total": total_code, "executed": executed}
    return stats


def check_local_markdown_links():
    link_pattern = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")
    errors = []
    checked = 0
    for md_file in find_files("*.md"):
        relative = md_file.relative_to(REPO_ROOT)
        content = md_file.read_text(encoding="utf-8")
        for match in link_pattern.finditer(content):
            url = match.group(2)
            if url.startswith(("http", "#", "mailto:")):
                continue
            checked += 1
            path = url.split("#", 1)[0]
            target = (md_file.parent / path).resolve()
            try:
                target_relative = target.relative_to(REPO_ROOT)
            except ValueError:
                errors.append(f"{relative}: ссылка '{match.group(2)}' вне репозитория")
                continue
            if not target.exists():
                errors.append(
                    f"{relative}: ссылка '{match.group(2)}' -> "
                    f"{target_relative} не найдена"
                )
    return errors, checked


def find_publishable_text_files():
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file() or any(excluded in path.parts for excluded in EXCLUDE_DIRS):
            continue
        if path.suffix.lower() in TEXT_SUFFIXES or path.name == ".gitignore":
            yield path


def check_sensitive_patterns():
    patterns = {
        "signed_url": re.compile(
            r"https?://[^\s<>]*[?&](?:X-Amz-Signature|Signature|sig|token|access_token|AWSAccessKeyId)=[^\s<>]*",
            re.IGNORECASE,
        ),
        "jwt": re.compile(r"eyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+"),
        "abs_local_path": re.compile(
            r"(?<!\w)/(?:home|Users|tmp|mnt|media|run)/[^\s\)\]\"'>]+|[A-Za-z]:\\Users\\[^\s\"'>]+"
        ),
    }
    findings = []
    for text_file in find_publishable_text_files():
        relative = text_file.relative_to(REPO_ROOT)
        try:
            content = text_file.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            findings.append(f"{relative}: [read_error] {error}")
            continue
        for line_number, line in enumerate(content.splitlines(), 1):
            for name, pattern in patterns.items():
                if pattern.search(line):
                    findings.append(f"{relative}:{line_number}: [{name}] {line.strip()[:120]}")
    return findings


def check_notebook_inventory():
    notebooks = {path.relative_to(REPO_ROOT) for path in find_files("*.ipynb")}
    errors = [f"отсутствует обязательный notebook: {path}" for path in sorted(EXPECTED_NOTEBOOKS - notebooks)]
    student = sum("solution" not in path.name for path in notebooks)
    solution = sum("solution" in path.name for path in notebooks)
    if student < 7 or solution < 7:
        errors.append(f"ожидалось не менее 7 студенческих и 7 эталонных notebooks, найдено {student} и {solution}")
    return student, solution, errors


def extract_external_urls():
    url_pattern = re.compile(r"https?://[^\s<>\]\"'`]+")
    urls = set()
    for md_file in find_files("*.md"):
        for match in url_pattern.finditer(md_file.read_text(encoding="utf-8")):
            url = match.group(0).rstrip(".,;:!?")
            while url.endswith(")") and url.count("(") < url.count(")"):
                url = url[:-1]
            urls.add(url)
    return sorted(urls)


def request_external_url(url):
    last_error = None
    for method in ("HEAD", "GET"):
        request = urllib.request.Request(url, method=method, headers={"User-Agent": "Mozilla/5.0"})
        if method == "GET":
            request.add_header("Range", "bytes=0-0")
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                return "ok", str(response.status)
        except Exception as error:
            last_error = error

    if isinstance(last_error, urllib.error.HTTPError):
        if last_error.code in {404, 410}:
            return "failed", f"HTTP {last_error.code}"
        return "warning", f"HTTP {last_error.code}"
    return "warning", str(last_error)


def run_external_link_check():
    urls = extract_external_urls()
    ok = 0
    failed = []
    warnings = []
    for url in urls:
        status, detail = request_external_url(url)
        if status == "ok":
            ok += 1
        elif status == "failed":
            failed.append(f"{url} -> {detail}")
        else:
            warnings.append(f"{url} -> {detail}")
    return len(urls), ok, failed, warnings


def print_errors(errors, prefix="ОШИБКА"):
    for error in errors:
        print(f"  {prefix}: {error}")


def main():
    parser = argparse.ArgumentParser(description="Валидация репозитория TOP_II_8")
    parser.add_argument("--check-external", action="store_true", help="Проверить внешние ссылки")
    args = parser.parse_args()

    print("=" * 60)
    print("Валидация репозитория TOP_II_8")
    print("=" * 60)
    print()
    all_ok = True

    print("--- 1. Python-синтаксис ---")
    py_errors = check_python_syntax()
    if py_errors:
        all_ok = False
        print_errors(py_errors)
    else:
        print(f"  {sum(1 for _ in find_files('*.py'))} файлов проверено, ошибок нет")
    print()

    print("--- 2. nbformat notebooks ---")
    nb_errors = check_notebook_format()
    if nb_errors:
        all_ok = False
        print_errors(nb_errors)
    else:
        print(f"  {sum(1 for _ in find_files('*.ipynb'))} notebooks прошли schema-validation")
    print()

    print("--- 3. Синтаксис code-ячеек ---")
    cell_errors = check_code_cell_syntax()
    if cell_errors:
        all_ok = False
        print_errors(cell_errors)
    else:
        total_code = sum(
            sum(cell.cell_type == "code" for cell in read_notebook(nb_file).cells)
            for nb_file in find_files("*.ipynb")
        )
        print(f"  {total_code} code-ячеек проверено, ошибок нет")
    print()

    print("--- 4. Error outputs ---")
    output_errors = check_error_outputs()
    if output_errors:
        all_ok = False
        print_errors(output_errors)
    else:
        print("  error outputs не найдены")
    print()

    print("--- 5. Исполненность code-ячеек эталонов ---")
    execution_stats = check_executed_solution_cells()
    all_executed = bool(execution_stats)
    total_solution_code = 0
    total_solution_executed = 0
    for relative, stats in sorted(execution_stats.items()):
        total_solution_code += stats["total"]
        total_solution_executed += stats["executed"]
        complete = stats["executed"] == stats["total"]
        all_executed &= complete
        label = "OK" if complete else "НЕ ПОЛНОСТЬЮ"
        print(f"  {label}: {relative}: {stats['executed']}/{stats['total']}")
    if not all_executed:
        all_ok = False
    print(f"  Итого: {total_solution_executed}/{total_solution_code} code-ячеек эталонов исполнено")
    print()

    print("--- 6. Локальные Markdown-ссылки ---")
    link_errors, link_count = check_local_markdown_links()
    if link_errors:
        all_ok = False
        print_errors(link_errors)
    else:
        print(f"  {link_count} ссылок проверено, ошибок нет")
    print()

    print("--- 7. Конфиденциальные данные и локальные пути ---")
    sensitive_findings = check_sensitive_patterns()
    if sensitive_findings:
        all_ok = False
        print_errors(sensitive_findings, prefix="НАЙДЕНО")
    else:
        print("  подписанные URL, JWT и абсолютные локальные пути не найдены")
    print()

    print("--- 8. Состав notebooks ---")
    student_notebooks, solution_notebooks, inventory_errors = check_notebook_inventory()
    print(f"  Студенческих: {student_notebooks}")
    print(f"  Эталонных: {solution_notebooks}")
    print(f"  Всего: {student_notebooks + solution_notebooks}")
    if inventory_errors:
        all_ok = False
        print_errors(inventory_errors)
    print()

    if args.check_external:
        print("--- 9. Внешние ссылки (--check-external) ---")
        checked, ok, failed, warnings = run_external_link_check()
        if failed:
            all_ok = False
            print_errors(failed, prefix="НЕДОСТУПНА")
        print_errors(warnings, prefix="СЕТЕВОЕ ПРЕДУПРЕЖДЕНИЕ")
        print(f"  Уникальных URL: {checked}, доступно: {ok}, ошибок: {len(failed)}, предупреждений: {len(warnings)}")
        print()

    print("=" * 60)
    if all_ok:
        print("РЕЗУЛЬТАТ: все проверки пройдены")
    else:
        print("РЕЗУЛЬТАТ: обнаружены ошибки (см. выше)")
        sys.exit(1)


if __name__ == "__main__":
    main()

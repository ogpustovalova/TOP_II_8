"""Сборка Jupyter-ноутбуков для КИМ курса «Основы нейронных сетей».

Ноутбуки строятся из списка ячеек в формате эталона examples/lab-regression-features.ipynb:
- markdown-ячейки с теорией и инструкциями;
- code-ячейки с плейсхолдером `# < ENTER YOUR CODE HERE >` для студентов.

Использование:
    from nb_builder import Notebook, md, code, placeholder, task

    nb = Notebook("Название КИМ")
    nb.add(md("# Заголовок"))
    nb.add(md("Инструкция..."))
    nb.add(placeholder())  # пустая ячейка для кода студента
    nb.save("path/to/kim-XX-name.ipynb")
"""
import json
import uuid
from pathlib import Path


_id_counter = 0


def _next_id() -> str:
    """Детерминированный id ячейки (читаемый, без зависимости от окружения)."""
    global _id_counter
    _id_counter += 1
    return f"cell-{_id_counter:03d}"


def md(text: str):
    """Markdown-ячейка. text может содержать переводы строк."""
    return {"cell_type": "markdown", "source": _split(text), "metadata": {},
            "id": _next_id()}


def code(text: str = ""):
    """Code-ячейка с произвольным содержимым."""
    return {"cell_type": "code", "source": _split(text), "metadata": {},
            "execution_count": None, "outputs": [], "id": _next_id()}


def placeholder(comment: str = ""):
    """Пустая code-ячейка с плейсхолдером для кода студента.

    Формат идентичен эталону examples/lab-regression-features.ipynb.
    """
    src = "# < ENTER YOUR CODE HERE >"
    if comment:
        src += f"  # {comment}"
    return code(src)


def task(text: str):
    """Markdown-ячейка с заданием (короткое описание шага)."""
    return md(text)


def _split(text: str):
    """Разбить текст на строки в формате ipynb: каждая строка, кроме последней,
    оканчивается '\\n'. ipynb хранит source как list строк."""
    if not text:
        return []
    lines = text.split("\n")
    # Все строки, кроме последней, должны оканчиваться на \n
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


class Notebook:
    """Сборщик Jupyter-ноутбука."""

    def __init__(self, title: str = ""):
        self.title = title
        self.cells: list[dict] = []
        global _id_counter
        _id_counter = 0  # нумерация id с 001 в каждом ноутбуке

        # Язык ядра — Python 3
        self.metadata = {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.x",
                "mimetype": "text/x-python",
                "file_extension": ".py",
                "pygments_lexer": "ipython3",
            },
        }
        self.nbformat = 4
        self.nbformat_minor = 5

    def add(self, cell: dict):
        """Добавить ячейку."""
        self.cells.append(cell)
        return self

    def add_many(self, cells: list[dict]):
        """Добавить несколько ячеек."""
        self.cells.extend(cells)
        return self

    def build(self) -> dict:
        """Собрать структуру ноутбука как dict."""
        return {
            "cells": self.cells,
            "metadata": self.metadata,
            "nbformat": self.nbformat,
            "nbformat_minor": self.nbformat_minor,
        }

    def save(self, path: str):
        """Сохранить ноутбук в файл."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.build(), f, ensure_ascii=False, indent=1)
        return p

    def cell_count(self) -> int:
        return len(self.cells)

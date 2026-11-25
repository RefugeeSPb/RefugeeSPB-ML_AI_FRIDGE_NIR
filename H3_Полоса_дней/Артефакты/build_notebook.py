"""Собрать H3_результаты.ipynb с таблицами и картинками внутри ячеек."""

from __future__ import annotations

import base64
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from nbformat import writes
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook, new_output

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "результаты"
DATA = HERE / "данные"
NOTEBOOK = HERE / "H3_результаты.ipynb"


def png_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def image_output(path: Path):
    return new_output(
        "display_data",
        data={"image/png": png_b64(path), "text/plain": [path.name]},
        metadata={"image/png": {"width": 720}},
    )


def table_output(frame: pd.DataFrame, count: int):
    return new_output(
        "execute_result",
        data={
            "text/plain": [frame.to_string(index=False)],
            "text/html": [frame.to_html(index=False)],
        },
        metadata={},
        execution_count=count,
    )


def figure_cell(path: Path, filename: str, comment: str, count: int):
    code = new_code_cell(
        f"{comment}\n"
        f"display(Image(str(RESULTS / {filename!r}), width=720))"
    )
    code["outputs"] = [image_output(path)]
    code["execution_count"] = count
    return code


def bar_cover() -> Path:
    path = RESULTS / "покрытие.png"
    labels = ["полоса ошибок", "окно по F_d"]
    values = [0.794, 0.827]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(labels, values, color=["#9aa4b2", "#123194"])
    ax.axhline(0.78, color="#c0392b", linestyle="--", linewidth=1, label="порог 0,78")
    ax.set_ylabel("покрытие")
    ax.set_ylim(0.7, 0.9)
    ax.set_title("H3. Семя 44, 500 семей")
    ax.legend()
    for name, value in zip(labels, values):
        ax.text(name, value + 0.008, f"{value:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def bar_width() -> Path:
    path = RESULTS / "ширина.png"
    labels = ["полоса ошибок", "окно по F_d"]
    values = [5.59, 4.38]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(labels, values, color=["#9aa4b2", "#123194"])
    ax.set_ylabel("средняя ширина, дни")
    ax.set_ylim(0, 7)
    ax.set_title("Короче на 21,6%")
    for name, value in zip(labels, values):
        ax.text(name, value + 0.12, f"{value:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def calendar_example() -> Path:
    path = RESULTS / "пример_календаря.png"
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    days = list(range(1, 15))
    ax.set_xlim(0.5, 14.5)
    ax.set_ylim(0, 3)
    ax.set_yticks([1, 2])
    ax.set_yticklabels(["гречка", "молоко"])
    ax.set_xticks(days)
    ax.set_xlabel("день после последней покупки")
    ax.barh(2, 6, left=4, height=0.45, color="#123194", alpha=0.85)
    ax.barh(1, 5, left=8, height=0.45, color="#123194", alpha=0.85)
    ax.axvline(6, color="#9aa4b2", linestyle=":", linewidth=1)
    ax.text(6.15, 2.55, "точка «6-й день»", color="#9aa4b2", fontsize=8)
    ax.set_title("Свои дни у продукта, не один шаблон")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(RESULTS / "полоса.csv") if (RESULTS / "полоса.csv").exists() else pd.read_csv(DATA / "метрики" / "полоса_h3b.csv")
    extra = pd.read_csv(DATA / "метрики" / "полоса_справка.csv")
    cover = bar_cover()
    width = bar_width()
    calendar = calendar_example()

    cells = [
        new_markdown_cell(
            "# H3. Полоса дней — результаты\n\n"
            "## Сокращения, аббревиатуры и метрики\n\n"
            "| Как написано | Расшифровка |\n"
            "|---|---|\n"
            "| **H3** | гипотеза 3: короткое окно дней по продукту. |\n"
            "| **окно**, полоса | дни, в которые ждут повторную покупку. |\n"
            "| **покрытие** | доля строк, где настоящий день внутри окна. |\n"
            "| **ширина** | правый край минус левый, среднее по строкам, в днях. |\n"
            "| **сужение** | на сколько окно короче полосы ошибок: "
            "(ширина ошибок − ширина F_d) / ширина ошибок. "
            "0,216 = короче на 21,6%. |\n"
            "| **полоса ошибок** | одна длина из прошлых промахов точечного прогноза. |\n"
            "| **F_d** | к дню d покупка уже случилась. Края окна: 0,1 и 0,9. |\n"
            "| **h_d** | купят в день d, если раньше не купили. |\n"
            "| **MLP** | сеть, слои 256 и 128, 14 логитов дней. |\n"
            "| **CatBoost квантили** | две модели на 0,1 и 0,9. Не H3, порог не взят. |\n"
            "| **семя 44** | генератор официальной базы 500 семей. |\n"
            "| **SQLite**, `fridge_lab_v3` | эта база. Старые 200 и семя 43 в вердикт не входят. |\n\n"
            "Гипотеза: окно по F_d накрывает день **не реже 78 из 100** "
            "и **короче** полосы ошибок **не меньше чем на 10%**.\n\n"
            "Ноутбук **не учит** сеть. Повтор — `код/interval_h3b.py` на "
            "`fridge_lab_v3.sqlite`. Вердикт без обучения — `код/eval_h3.py`.\n\n"
            "Порядок этапов:\n"
            "1. сводная таблица порога;\n"
            "2. покрытие;\n"
            "3. ширина;\n"
            "4. как это выглядит у молока;\n"
            "5. старые постановки, не вердикт."
        ),
        new_markdown_cell(
            "## Этап 1. Сводная таблица порога\n\n"
            "Читаем `результаты/полоса.csv`. Одна строка — официальный тест.\n\n"
            "- **ошибки** — полоса одной длины вокруг точки.\n"
            "- **F_d** — окно с краёв 0,1 и 0,9 накопленной доли.\n"
            "- Покрытие ≥ 0,78 и сужение ≥ 10% записаны до сборки базы семя 44."
        ),
        new_code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "from IPython.display import Image, display\n"
            "\n"
            "HERE = Path.cwd()\n"
            "if not (HERE / \"результаты\").is_dir():\n"
            "    HERE = Path(\".\").resolve()\n"
            "RESULTS = HERE / \"результаты\"\n"
            "DATA = HERE / \"данные\" / \"метрики\"\n"
            "\n"
            "pd.read_csv(RESULTS / \"полоса.csv\")"
        ),
    ]
    cells[-1]["outputs"] = [table_output(summary, 1)]
    cells[-1]["execution_count"] = 1

    cells.append(
        new_markdown_cell(
            "## Этап 2. Покрытие, семя 44\n\n"
            "237 271 строка. Полоса ошибок: 0,794. Окно по F_d: **0,827**. "
            "Красная черта — порог 0,78. День внутри в 83 случаях из 100 против 79."
        )
    )
    cells.append(
        figure_cell(cover, "покрытие.png", "# Официальный тест: день внутри окна.", 2)
    )

    cells.append(
        new_markdown_cell(
            "## Этап 3. Ширина\n\n"
            "Ошибки: 5,59 дня. F_d: **4,38 дня**. Короче на **21,6%**, "
            "порог 10%. 4,38 — средняя длина окна, не «купят за четыре дня»."
        )
    )
    cells.append(
        figure_cell(width, "ширина.png", "# Короче полосы ошибок при покрытии выше порога.", 3)
    )

    cells.append(
        new_markdown_cell(
            "## Этап 4. Молоко и гречка\n\n"
            "Схема, не одна строка теста. Точка «6-й день» одна. "
            "Окно молока ближе, окно гречки позже. На экране — эти дни, "
            "не «в пятницу шанс 10%»."
        )
    )
    cells.append(
        figure_cell(
            calendar,
            "пример_календаря.png",
            "# Свои дни у продукта. 10% и 90% — края расчёта.",
            4,
        )
    )

    cells.append(
        new_markdown_cell(
            "## Этап 5. Что не входит в вердикт\n\n"
            "Квантили CatBoost и старый коридор 78–82% на базах семя 42 и 43 "
            "смотрели раньше. Порог H3 записан до базы семя 44. Эти строки — справка."
        )
    )
    extra_cell = new_code_cell(
        "# Старые постановки. В порог H3 не входят.\n"
        "pd.read_csv(DATA / \"полоса_справка.csv\")"
    )
    extra_cell["execution_count"] = 5
    extra_cell["outputs"] = [table_output(extra, 5)]
    cells.append(extra_cell)

    nb = new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
    )
    NOTEBOOK.write_text(writes(nb), encoding="utf-8")
    print("wrote", NOTEBOOK)


if __name__ == "__main__":
    main()

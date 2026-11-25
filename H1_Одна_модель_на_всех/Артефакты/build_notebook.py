"""Собрать H1_результаты.ipynb с таблицами и картинками внутри ячеек."""

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
NOTEBOOK = HERE / "H1_результаты.ipynb"


def png_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def image_output(path: Path):
    return new_output(
        "display_data",
        data={"image/png": png_b64(path), "text/plain": [path.name]},
        metadata={"image/png": {"width": 720}},
    )


def load_households() -> pd.DataFrame:
    raw = pd.read_csv(DATA / "метрики" / "forecast_v2_households.csv", header=None)
    body = raw.iloc[3:].copy()
    body.columns = [
        "hh",
        "freq_r",
        "freq_p",
        "glob_r",
        "glob_p",
        "pers_r",
        "pers_p",
        "sim_r",
        "sim_p",
        "corr_r",
        "corr_p",
    ]
    for column in body.columns:
        body[column] = pd.to_numeric(body[column])
    return body.reset_index(drop=True)


def bar_official(personal: float, general: float) -> Path:
    path = RESULTS / "recall10_сравнение.png"
    labels = ["только эта семья", "общая модель"]
    values = [personal, general]
    colors = ["#9aa4b2", "#123194"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(labels, values, color=colors)
    ax.axhline(personal, color="#c0392b", linestyle="--", linewidth=1, label="личная: общая лучше, если выше")
    ax.set_ylabel("Recall@10, среднее по семьям")
    ax.set_ylim(0, 0.85)
    ax.set_title("H1. Семя 42, 200 семей")
    ax.legend()
    for name, value in zip(labels, values):
        ax.text(name, value + 0.015, f"{value:.3f}", ha="center", fontsize=9)
    ax.text(0.5, 0.78, f"+{general - personal:.3f}", ha="center", color="#123194", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def scatter_gain(frame: pd.DataFrame) -> Path:
    path = RESULTS / "personal_vs_global.png"
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(frame["pers_r"], frame["glob_r"], s=18, alpha=0.7, color="#123194")
    lo = min(frame["pers_r"].min(), frame["glob_r"].min()) - 0.02
    hi = max(frame["pers_r"].max(), frame["glob_r"].max()) + 0.02
    ax.plot([lo, hi], [lo, hi], color="#9aa4b2", linewidth=1)
    ax.set_xlabel("только эта семья")
    ax.set_ylabel("общая модель")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title("Каждая точка — одна семья")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def history_plot() -> Path:
    path = RESULTS / "history_length.png"
    hist = pd.read_csv(DATA / "метрики" / "forecast_v2_history_length.csv")
    keep = hist[hist["Недель истории"] != "вся история"].copy()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(
        keep["Недель истории"],
        keep["Личная модель"],
        marker="o",
        color="#9aa4b2",
        label="только эта семья",
    )
    ax.plot(
        keep["Недель истории"],
        keep["Общая модель"],
        marker="o",
        color="#123194",
        label="общая модель",
    )
    ax.set_xlabel("недель своей истории")
    ax.set_ylabel("Recall@10")
    ax.set_ylim(0.4, 0.7)
    ax.set_title("40 семей, история обрезана")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


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


def main() -> None:
    frame = load_households()
    personal = float(frame["pers_r"].mean())
    general = float(frame["glob_r"].mean())
    official = bar_official(personal, general)
    scatter = scatter_gain(frame)
    history = history_plot()
    summary = pd.read_csv(RESULTS / "recall10.csv")
    old_scatter = RESULTS / "personal_vs_correction.png"
    if old_scatter.exists():
        old_scatter.unlink()

    cells = [
        new_markdown_cell(
            "# H1. Одна модель на всех — результаты\n\n"
            "## Сокращения, аббревиатуры и метрики\n\n"
            "| Как написано | Расшифровка |\n"
            "|---|---|\n"
            "| **H1** | гипотеза 1: одна общая модель лучше модели только на этой семье. |\n"
            "| **десятка**, top-10 | первые 10 продуктов в списке кандидатов. |\n"
            "| **Recall@10** | полнота десятки внутри семьи, затем среднее по семьям. "
            "Каждая семья весит одинаково. |\n"
            "| **личная модель** | CatBoost только на покупках этой семьи, с нуля. |\n"
            "| **общая модель** | один CatBoost на всех семьях. Частота этой семьи "
            "в признаках уже есть. |\n"
            "| **лучше** | низ 95% интервала «общая − личная» строго выше нуля. |\n"
            "| **поправка** | общая плюс добавка по семье. Предыдущая постановка, "
            "в вердикт H1 не входит. |\n"
            "| **CI**, 95% интервал | бутстреп по семьям, 2000 повторов, семя 42. |\n"
            "| **seed / семя** | 42: генератор учебной базы, CatBoost и бутстреп. |\n"
            "| **CatBoost** | градиентный бустинг на деревьях с категориями. |\n"
            "| **SQLite**, `fridge_lab` | 200 учебных семей, семя генератора 42. |\n\n"
            "Гипотеза: общая модель угадывает десятку **лучше**, чем модель "
            "только на этой семье.\n\n"
            "Ноутбук **не учит** модели. Он читает `данные/метрики/`. "
            "Повтор обучения — `код/personal_h1.py` на `fridge_lab.sqlite`. "
            "Вердикт без обучения — `код/eval_h1.py`.\n\n"
            "Порядок этапов:\n"
            "1. пути и сводная таблица;\n"
            "2. официальный результат, семя 42;\n"
            "3. каждая семья — точка;\n"
            "4. короткая история;\n"
            "5. что в столбцах csv."
        ),
        new_markdown_cell(
            "## Этап 1. Пути и сводная таблица\n\n"
            "Читаем `результаты/recall10.csv`. Строка — одна база, не один поход.\n\n"
            "- **только_семья** — CatBoost на строках этой семьи.\n"
            "- **общая** — один CatBoost на всех семьях.\n"
            "- **прирост** — разница Recall@10, среднее по 200 семьям. "
            "Общая лучше, если низ интервала > 0. Семя 42.\n\n"
            "Поправка к общей модели в таблицу вердикта не входит. "
            "«Лучше на 0,02» — другая постановка, сюда не входит."
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
            "pd.read_csv(RESULTS / \"recall10.csv\")"
        ),
    ]
    cells[-1]["outputs"] = [table_output(summary, 1)]
    cells[-1]["execution_count"] = 1

    cells.append(
        new_markdown_cell(
            "## Этап 2. Официальный результат, семя 42\n\n"
            "200 семей. Только эта семья: 0,612. Общая: 0,625. "
            "Прирост **+0,013**. Красная черта — уровень личной: "
            "общая лучше, если выше.\n\n"
            "Интервал [0,009; 0,018] целиком выше нуля. Вердикт — "
            "лучше, подтверждена."
        )
    )
    cells.append(
        figure_cell(
            official,
            "recall10_сравнение.png",
            "# Сводка для защиты: одна база, семя 42, лучше если низ > 0.",
            2,
        )
    )

    cells.append(
        new_markdown_cell(
            "## Этап 3. Каждая семья — точка\n\n"
            "По горизонтали — модель только на этой семье. По вертикали — "
            "общая. Выше диагонали общая выиграла.\n\n"
            "134 семьи из 200 выше диагонали. Есть семьи, где своя модель лучше: "
            "гипотеза смотрит среднее, не «всем одинаково»."
        )
    )
    cells.append(
        figure_cell(
            scatter,
            "personal_vs_global.png",
            "# Точка — семья. Диагональ — равенство двух способов.",
            3,
        )
    )

    cells.append(
        new_markdown_cell(
            "## Этап 4. Короткая история\n\n"
            "У 40 семей историю обрезали до 3, 4, 6, 8 и 16 недель. "
            "Модель только на семье на короткой истории слабее. "
            "Общая видит всех и почти не падает.\n\n"
            "На полной истории общая тоже не ниже. "
            "График — не вердикт, а пояснение, почему учить только на семье рано."
        )
    )
    cells.append(
        figure_cell(
            history,
            "history_length.png",
            "# 40 семей: своя история обрезана, общая модель видит всех.",
            4,
        )
    )

    cells.append(
        new_markdown_cell(
            "## Этап 5. Что в csv семей\n\n"
            "В `forecast_v2_households.csv` каждая строка — семья. "
            "Для H1 нужны столбцы «Личная модель» и «Общая модель», "
            "метрика Recall@10.\n\n"
            "| Столбец | Роль в H1 |\n"
            "|---|---|\n"
            "| Личная модель | база: только эта семья |\n"
            "| Общая модель | вариант гипотезы |\n"
            "| Общая + личная поправка | предыдущая постановка, в критерий не входит |\n"
            "| Частота | правило H1, сюда не сравниваем |\n"
            "| Личная + похожие | другая постановка, не H1 |\n\n"
            "Ниже — первые пять семей, только столбцы вердикта."
        )
    )
    preview = frame[["hh", "pers_r", "glob_r"]].head(5).rename(
        columns={"hh": "семья", "pers_r": "только_семья", "glob_r": "общая"}
    )
    preview.to_csv(RESULTS / "households_head.csv", index=False)
    extra = new_code_cell(
        "# Первые семьи. Полный файл — данные/метрики/forecast_v2_households.csv.\n"
        "pd.read_csv(RESULTS / \"households_head.csv\")"
    )
    extra["execution_count"] = 5
    extra["outputs"] = [table_output(preview, 5)]
    cells.append(extra)

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

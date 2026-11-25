"""Собрать H2_результаты.ipynb с таблицами и картинками внутри ячеек."""

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
NOTEBOOK = HERE / "H2_результаты.ipynb"


def png_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def image_output(path: Path):
    return new_output(
        "display_data",
        data={"image/png": png_b64(path), "text/plain": [path.name]},
        metadata={"image/png": {"width": 720}},
    )


def bar_official() -> Path:
    path = RESULTS / "recall10_сравнение.png"
    labels = ["200 семей", "500 семей"]
    freq = [0.494, 0.496]
    model = [0.631, 0.648]
    x = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar([i - width / 2 for i in x], freq, width, label="обычно берут", color="#9aa4b2")
    ax.bar([i + width / 2 for i in x], model, width, label="модель с остатком", color="#123194")
    ax.axhline(0.10, color="#cccccc", linewidth=0)
    ax.set_ylabel("Recall@10")
    ax.set_xticks(list(x), labels)
    ax.set_ylim(0, 0.85)
    ax.set_title("H2. Список к походу")
    ax.legend()
    for i, (a, b) in enumerate(zip(freq, model)):
        ax.text(i - width / 2, a + 0.015, f"{a:.3f}", ha="center", fontsize=9)
        ax.text(i + width / 2, b + 0.015, f"{b:.3f}", ha="center", fontsize=9)
        ax.text(i, 0.78, f"+{b - a:.3f}", ha="center", color="#123194", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def bar_rules() -> Path:
    path = RESULTS / "recall10_правила_тест.png"
    metrics = pd.read_csv(DATA / "метрики" / "forecast_v1_metrics.csv")
    test = metrics[metrics["split"] == "test"].copy()
    keep = [
        "Доля походов с продуктом",
        "Личный интервал",
        "Запас в приложении",
        "Привычный и кончается",
        "CatBoost: след. поход",
    ]
    test = test[test["Модель"].isin(keep)]
    test["Модель"] = test["Модель"].replace(
        {
            "Доля походов с продуктом": "обычно берут",
            "CatBoost: след. поход": "модель с остатком",
        }
    )
    order = ["обычно берут", "Личный интервал", "Запас в приложении", "Привычный и кончается", "модель с остатком"]
    test["Модель"] = pd.Categorical(test["Модель"], order, ordered=True)
    test = test.sort_values("Модель")
    colors = ["#9aa4b2", "#b0b8c4", "#c5ccd4", "#7a8796", "#123194"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(test["Модель"].astype(str), test["Recall@10"], color=colors)
    ax.set_ylabel("Recall@10, тест 200 семей")
    ax.set_ylim(0, 0.85)
    ax.set_title("Те же походы, разные правила")
    ax.tick_params(axis="x", rotation=20)
    for name, value in zip(test["Модель"].astype(str), test["Recall@10"]):
        ax.text(name, value + 0.015, f"{value:.3f}", ha="center", fontsize=9)
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
    official = bar_official()
    rules = bar_rules()
    summary = pd.read_csv(RESULTS / "recall10.csv")
    test = pd.read_csv(DATA / "метрики" / "forecast_v1_metrics.csv")
    test = test[test["split"] == "test"][["Модель", "Recall@10", "HitRate@10", "PR-AUC след. поход"]]

    cells = [
        new_markdown_cell(
            "# H2. Список к походу — результаты\n\n"
            "## Сокращения, аббревиатуры и метрики\n\n"
            "| Как написано | Расшифровка |\n"
            "|---|---|\n"
            "| **H2** | гипотеза 2 этой предзащиты: список к походу. |\n"
            "| **десятка**, top-10 | первые 10 продуктов в отсортированном списке кандидатов. |\n"
            "| **Recall@10** | полнота десятки. Сколько купленного в *следующем* походе попало "
            "в десятку, делённое на размер этой корзины. Среднее по походам. "
            "Лишние советы число не уменьшают. Главная метрика H2. |\n"
            "| **HitRate@10** | доля походов, где в десятку попал *хотя бы один* купленный продукт. "
            "Грубее Recall@10: не смотрит, сколько угадали. |\n"
            "| **PR-AUC** | Precision–Recall Area Under Curve, площадь под кривой "
            "точность–полнота. Считается по вероятности «будет в следующей корзине» "
            "на всех позициях, не только на десятке. |\n"
            "| **CatBoost** | Categorical Boosting: градиентный бустинг на решающих деревьях "
            "с категориями (код продукта, категория, тип семьи). Статья: "
            "Prokhorenkova et al., NeurIPS 2018. |\n"
            "| **нейросеть**, MLP | многослойный перцептрон. В таблице этапа 7 — соседняя "
            "постановка, не вариант вердикта H2. |\n"
            "| **обычно берут**, `pair_visit_share` | доля походов семьи, в которых продукт "
            "уже брали: (число таких походов) / (число походов семьи к этому моменту). "
            "Правило сравнения H2. |\n"
            "| **split** | разрез данных по времени: train — учёба, validation — ранняя "
            "остановка, test — вердикт. |\n"
            "| **Ta Feng** | открытые тайваньские чеки 2000–2001. Остатка холодильника нет. "
            "Отрицательный контроль H2. |\n"
            "| **SQLite**, `fridge_lab` | учебная база синтетических семей с остатком и планом "
            "похода. Файл прогона — `fridge_lab_v2.sqlite`. |\n"
            "| **CSV** | таблица уже посчитанных метрик. Ноутбук её только читает. |\n\n"
            "Гипотеза: модель, которая видит **остаток, расход и срок**, "
            "собирает десятку к следующему походу лучше правила «обычно берут» "
            "не меньше чем на **0,10 Recall@10**.\n\n"
            "Ноутбук **не учит** CatBoost. Он показывает уже посчитанные таблицы и картинки "
            "из папок `результаты/` и `данные/метрики/`. Повтор прогона — "
            "`код/check_main_v2.py` на `fridge_lab_v2.sqlite`.\n\n"
            "Порядок этапов:\n"
            "1. пути и сводная таблица порога;\n"
            "2. официальный результат 200 / 500 семей;\n"
            "3. те же походы, простые правила;\n"
            "4. пример одной десятки;\n"
            "5. разброс по семьям;\n"
            "6. запас как шумный вход;\n"
            "7. полная таблица теста и что стоит в столбце «Модель»."
        ),
        new_markdown_cell(
            "## Этап 1. Пути и сводная таблица порога\n\n"
            "Читаем `результаты/recall10.csv`. Строка — одна база, не один поход.\n\n"
            "- **обычно_берут** — сортировка кандидатов по доле походов семьи с продуктом "
            "(`pair_visit_share`).\n"
            "- **модель** — CatBoost на истории, остатке, расходе, сроке и днях до похода.\n"
            "- **прирост** — разница Recall@10. Порог 0,10 записан до теста и здесь не двигается.\n\n"
            "Ta Feng — чужие чеки **без остатка**. Там порог не взят: без холодильника гипотеза "
            "на продукт не переносится."
        ),
        new_code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "from IPython.display import Image, display\n"
            "\n"
            "# Запускать из папки Артефакты/, где лежат результаты/ и данные/.\n"
            "HERE = Path.cwd()\n"
            "if not (HERE / \"результаты\").is_dir():\n"
            "    HERE = Path(\".\").resolve()\n"
            "RESULTS = HERE / \"результаты\"\n"
            "DATA = HERE / \"данные\" / \"метрики\"\n"
            "\n"
            "# Официальный вердикт H2: две учебные базы и отрицательный контроль без запаса.\n"
            "pd.read_csv(RESULTS / \"recall10.csv\")"
        ),
    ]
    cells[-1]["outputs"] = [table_output(summary, 1)]
    cells[-1]["execution_count"] = 1

    cells.append(
        new_markdown_cell(
            "## Этап 2. Официальный результат на учебной базе\n\n"
            "Слева частота («обычно берут»), справа модель с остатком. "
            "Над столбиками — прирост.\n\n"
            "200 семей, 2 807 походов: 0,494 → 0,631, **+0,137**. "
            "Повтор 500 семей, семя генератора 43, 6 646 походов: 0,496 → 0,648, **+0,152**. "
            "Оба раза порог 0,10 взят. Учёба до 1 июня 2026, тест 1 августа — 16 сентября, "
            "прогноз через 3 часа после уже сделанного похода."
        )
    )
    cells.append(
        figure_cell(
            official,
            "recall10_сравнение.png",
            "# Сводка для защиты: две учебные базы, один порог.",
            2,
        )
    )

    cells.append(
        new_markdown_cell(
            "## Этап 3. Те же походы, разные правила\n\n"
            "На тесте 200 семей частота — лучшее **простое** правило (Recall@10 ≈ 0,494). "
            "Личный интервал и запас в приложении слабее. "
            "Правило «привычный и кончается» лучше чистого запаса, но до частоты не дотягивает "
            "(≈ 0,447).\n\n"
            "Модель на тех же кандидатах даёт ≈ 0,631. Значит выигрыш не от «просто посмотреть "
            "остаток», а от совместных признаков. Картинка строится из "
            "`данные/метрики/forecast_v1_metrics.csv`, split = test."
        )
    )
    cells.append(
        figure_cell(
            rules,
            "recall10_правила_тест.png",
            "# Контроль: частота, интервал, запас, «привычный и кончается», CatBoost.",
            3,
        )
    )

    cells.append(
        new_markdown_cell(
            "## Этап 4. Пример одной десятки\n\n"
            "Один поход одной семьи. Кандидаты отсортированы моделью сверху вниз. "
            "Зелёное — продукт реально купили в следующем походе, серое — нет. "
            "Черта отсекает топ-10: в Recall@10 входят только зелёные выше черты.\n\n"
            "Так видно, за счёт чего растёт метрика: в десятку попадает то, что кончается, "
            "а вчерашнее молоко с запасом уходит ниже."
        )
    )
    cells.append(
        figure_cell(
            RESULTS / "shopping_list.png",
            "shopping_list.png",
            "# Пример списка: зелёное купили, серое нет, черта — граница десятки.",
            4,
        )
    )

    cells.append(
        new_markdown_cell(
            "## Этап 5. Устойчивость по семьям\n\n"
            "Единица порога — поход, но разброс смотрим по семьям: среднее Recall@10 модели "
            "внутри семьи. Медиана около 0,62, типичный размах примерно 0,37…0,94.\n\n"
            "Гипотеза не «всем одинаково +0,14», а «в среднем по походам порог взят». "
            "Хвост слева — семьи, где десятка всё ещё слабая."
        )
    )
    cells.append(
        figure_cell(
            RESULTS / "household_recall.png",
            "household_recall.png",
            "# Распределение Recall@10 модели по семьям, не по походам.",
            5,
        )
    )

    cells.append(
        new_markdown_cell(
            "## Этап 6. Запас — вход модели, не правда холодильника\n\n"
            "Остаток в приложении шумит: пользователь отмечает покупку не сразу, "
            "списание идёт грубым правилом. На этапе 3 одно это правило проигрывает частоте.\n\n"
            "В модель запас всё равно входит вместе с историей, расходом и сроком. "
            "График — иллюстрация входа, не доказательство, что учёт «идеальный»."
        )
    )
    cells.append(
        figure_cell(
            RESULTS / "stock_timeline.png",
            "stock_timeline.png",
            "# Шумный ряд остатка: вход признака, не эталон холодильника.",
            6,
        )
    )

    cells.append(
        new_markdown_cell(
            "## Этап 7. Полная таблица теста, 200 семей\n\n"
            "Те же походы, что на этапе 3. Столбец **Модель** — способ отсортировать "
            "один и тот же список кандидатов (продукты семьи за 60 дней). "
            "HitRate@10 высокий почти у всех: в десятку хоть что-то из корзины попадает часто. "
            "Различает методы Recall@10. PR-AUC смотрит вероятности по всем позициям.\n\n"
            "### Что в столбце «Модель»\n\n"
            "Первые пять строк — **правила без обучения**. Дальше — обученные модели "
            "из того же прогона `forecast_v1_metrics.csv`.\n\n"
            "| Строка | Что это |\n"
            "|---|---|\n"
            "| **Доля походов с продуктом** | Правило «обычно берут»: `pair_visit_share`. "
            "База H2. Recall@10 ≈ 0,494. |\n"
            "| **Личный интервал** | Сколько дней осталось до обычного перерыва этой пары "
            "(средний интервал минус дни с последней покупки). Чем меньше осталось, "
            "тем выше в списке. |\n"
            "| **Запас в приложении** | На сколько дней хватит остатка при расходе "
            "за 30 дней. Чем меньше запас, тем выше. |\n"
            "| **Запас к походу** | Запас в днях минус дни до планового похода: "
            "«кончится до похода — купить». |\n"
            "| **Привычный и кончается** | «Запас к походу», но продукты с меньше чем "
            "тремя покупками уходят в конец. |\n"
            "| **CatBoost: след. поход** | **Вариант гипотезы H2.** Один CatBoost, "
            "метка «продукт будет в *следующей* корзине» (`in_next_basket`). "
            "Признаки: история пары, остаток, расход, срок, дни до похода; "
            "категории — продукт, категория, тип семьи. Сортирует тех же кандидатов, "
            "что и правило частоты. Recall@10 ≈ 0,631, прирост ≈ +0,137. "
            "Порог 0,10 берётся с этой строки. |\n"
            "| **CatBoost: 14 дней** | Тот же бустинг и те же признаки, но метка другая: "
            "«купят в ближайшие 14 дней», не «в следующем походе». "
            "Список к походу у неё хуже (≈ 0,585). В вердикт H2 не входит. |\n"
            "| **Нейросеть: 14 дней** | Многослойный перцептрон (два скрытых слоя), "
            "вероятность покупки за 14 дней. Соседняя постановка, не H2. |\n"
            "| **Нейросеть: к дате похода** | Та же сеть, вероятность к дате планового "
            "похода. Recall@10 близкий к H2 (≈ 0,635), PR-AUC ниже, чем у "
            "«CatBoost: след. поход». Порог H2 на неё не опирается. |\n\n"
            "Итог по столбцу: для H2 сравнивают только **Доля походов с продуктом** "
            "и **CatBoost: след. поход**. Остальные строки показывают, что выигрыш "
            "не от «просто посмотреть остаток» и не от любой модели с любой меткой."
        )
    )
    extra = new_code_cell(
        "# Только тест. Валидация в том же csv, в вердикт H2 не входит.\n"
        "metrics = pd.read_csv(DATA / \"forecast_v1_metrics.csv\")\n"
        "metrics[metrics[\"split\"] == \"test\"][\n"
        "    [\"Модель\", \"Recall@10\", \"HitRate@10\", \"PR-AUC след. поход\"]\n"
        "]"
    )
    extra["execution_count"] = 7
    extra["outputs"] = [table_output(test, 7)]
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

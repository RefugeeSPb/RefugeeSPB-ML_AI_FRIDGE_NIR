"""Собирает fridge_lab.sqlite из schema.sql и справочников catalog/*.csv.

Использование:
    python build_db.py                 # fridge_lab.sqlite рядом со скриптом
    python build_db.py --db other.sqlite

База пересобирается с нуля. Если в каталоге есть ошибки, существующий файл
не трогается.
"""

import argparse
import csv
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CATALOG = HERE / "catalog"
DIMENSIONS = {"mass", "volume", "count"}


def read_csv(name):
    with open(CATALOG / name, encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def positive_number(value, what, errors):
    try:
        number = float(value)
    except ValueError:
        errors.append(f"{what}: «{value}» — не число")
        return None
    if number <= 0:
        errors.append(f"{what}: {value} — должно быть больше нуля")
        return None
    return number


def unique_names(rows, file_name, errors, key="name"):
    seen = set()
    for row in rows:
        name = row[key].strip()
        if not name:
            errors.append(f"{file_name}: пустое название")
        elif name in seen:
            errors.append(f"{file_name}: «{name}» встречается дважды")
        seen.add(name)


def load_catalog():
    """Читает каталог и проверяет ссылки, единицы и числа. Возвращает строки и ошибки."""
    errors = []
    units = read_csv("units.csv")
    categories = read_csv("categories.csv")
    places = read_csv("storage_places.csv")
    bases = read_csv("base_products.csv")
    products = read_csv("products.csv")

    for rows, name in [(units, "units.csv"), (categories, "categories.csv"),
                       (places, "storage_places.csv"), (bases, "base_products.csv"),
                       (products, "products.csv")]:
        unique_names(rows, name, errors)

    unit_by_name = {}
    for row in units:
        if row["dimension"] not in DIMENSIONS:
            errors.append(f"units.csv, {row['name']}: неизвестная мера «{row['dimension']}»")
        row["to_base"] = positive_number(row["to_base"], f"units.csv, {row['name']}", errors)
        unit_by_name[row["name"]] = row

    category_names = {row["name"] for row in categories}
    place_names = {row["name"] for row in places}
    base_by_name = {}
    for row in bases:
        where = f"base_products.csv, {row['name']}"
        if row["category"] not in category_names:
            errors.append(f"{where}: нет категории «{row['category']}»")
        if row["storage_place"] not in place_names:
            errors.append(f"{where}: нет места хранения «{row['storage_place']}»")
        if row["dimension"] not in DIMENSIONS:
            errors.append(f"{where}: неизвестная мера «{row['dimension']}»")
        if row["is_perishable"] not in {"0", "1"}:
            errors.append(f"{where}: is_perishable должно быть 0 или 1")
        shelf = positive_number(row["typical_shelf_days"], f"{where}, срок", errors)
        row["typical_shelf_days"] = int(shelf) if shelf else None
        base_by_name[row["name"]] = row

    with_products = set()
    for row in products:
        where = f"products.csv, {row['name']}"
        base = base_by_name.get(row["base_product"])
        unit = unit_by_name.get(row["unit"])
        if base is None:
            errors.append(f"{where}: нет базового продукта «{row['base_product']}»")
        if unit is None:
            errors.append(f"{where}: нет единицы «{row['unit']}»")
        if base and unit and base["dimension"] != unit["dimension"]:
            errors.append(f"{where}: единица «{row['unit']}» не подходит для меры {base['dimension']}")
        row["pack_size"] = positive_number(row["pack_size"], f"{where}, фасовка", errors)
        row["price"] = positive_number(row["price"], f"{where}, цена", errors)
        if row["pack_size"] and unit and unit["to_base"]:
            row["pack_size_base"] = row["pack_size"] * unit["to_base"]
        with_products.add(row["base_product"])

    for name in base_by_name.keys() - with_products:
        errors.append(f"base_products.csv, {name}: нет ни одной фасовки в products.csv")

    return {"units": units, "categories": categories, "places": places,
            "bases": bases, "products": products}, errors


def ids(connection, table):
    return {name: id_ for id_, name in connection.execute(f"select id, name from {table}")}


def fill(connection, catalog):
    connection.executemany(
        "insert into unit (name, dimension, to_base) values (?, ?, ?)",
        [(r["name"], r["dimension"], r["to_base"]) for r in catalog["units"]],
    )
    connection.executemany("insert into category (name) values (?)",
                           [(r["name"],) for r in catalog["categories"]])
    connection.executemany("insert into storage_place (name) values (?)",
                           [(r["name"],) for r in catalog["places"]])

    category_id, place_id, unit_id = ids(connection, "category"), ids(connection, "storage_place"), ids(connection, "unit")
    connection.executemany(
        """insert into base_product
           (name, id_category, dimension, typical_shelf_days, is_perishable, id_storage_place)
           values (?, ?, ?, ?, ?, ?)""",
        [(r["name"], category_id[r["category"]], r["dimension"], r["typical_shelf_days"],
          int(r["is_perishable"]), place_id[r["storage_place"]]) for r in catalog["bases"]],
    )

    base_id = ids(connection, "base_product")
    connection.executemany(
        """insert into product (id_base_product, name, pack_size, id_unit, pack_size_base, price)
           values (?, ?, ?, ?, ?, ?)""",
        [(base_id[r["base_product"]], r["name"], r["pack_size"], unit_id[r["unit"]],
          r["pack_size_base"], r["price"]) for r in catalog["products"]],
    )


def check(connection):
    """Целостность, внешние ключи и то, что каждое представление выполняется."""
    assert connection.execute("pragma integrity_check").fetchone()[0] == "ok"
    broken = connection.execute("pragma foreign_key_check").fetchall()
    assert not broken, f"нарушены внешние ключи: {broken[:5]}"

    objects = connection.execute(
        "select type, name from sqlite_master where type in ('table', 'view') "
        "and name not like 'sqlite_%' order by type, name"
    ).fetchall()
    print(f"{'Объект':<28} Строк")
    for kind, name in objects:
        count = connection.execute(f"select count(*) from {name}").fetchone()[0]
        print(f"{kind:<5} {name:<22} {count}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=HERE / "fridge_lab.sqlite")
    args = parser.parse_args()

    catalog, errors = load_catalog()
    if errors:
        print("Каталог не собран, исправьте ошибки:", file=sys.stderr)
        for error in errors:
            print("  -", error, file=sys.stderr)
        sys.exit(1)

    draft = args.db.with_suffix(".building")
    draft.unlink(missing_ok=True)
    connection = sqlite3.connect(draft)
    try:
        connection.executescript((HERE / "schema.sql").read_text(encoding="utf-8"))
        with connection:
            fill(connection, catalog)
        check(connection)
    except BaseException:
        connection.close()
        draft.unlink(missing_ok=True)
        raise
    connection.close()
    draft.replace(args.db)
    print("Готово:", args.db)


if __name__ == "__main__":
    main()

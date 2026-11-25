"""3) Recall@10 правила «доля походов с продуктом».

Балл — ``pair_visit_share``: сколько прошлых походов семьи уже
содержали продукт, делённое на число её походов к моменту прогноза.
Десятка — десять наибольших баллов. Recall@10 похода — доля
продуктов следующей корзины, попавших в эту десятку.
Итог — среднее по походам. Считается в ``cost_hd.recall_at_10``.
"""

import cost_hd


def recall_rule(frame):
    """Вернуть средний Recall@10 правила по походам frame.

    Походы без купленной позиции в среднее не входят.
    В знаменателе Recall одного похода — размер корзины,
    не 10.

    Args:
        frame (pandas.DataFrame): Строки теста с
            ``visit_id``, ``base_product``, ``in_next_basket``
            и ``pair_visit_share``.

    Returns:
        tuple: ``(recall, n)`` — среднее Recall@10 и число
        походов в среднем.

    Examples:
        recall, trips = recall_rule(test)
    """
    score = frame["pair_visit_share"].to_numpy()
    return cost_hd.recall_at_10(frame, score)

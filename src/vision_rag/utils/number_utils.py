"""
数字格式工具 (Number Utilities)

核心职责:
  1. 将标准答案数字生成多种搜索变体 (用于 PDF 逆映射)
  2. 数字格式归一化 (千位分隔符, 货币符号, 百分号等)

示例:
  generate_number_variants("1,280.0")
  → ["1,280.0", "1280.0", "1,280", "1280", "$1,280.0", "$1,280", "1,280.00"]
"""

import re


def generate_number_variants(number_str: str) -> list[str]:
    """
    为给定的数字字符串生成多种格式变体，用于在 PDF 中搜索匹配。

    金融文档中同一个数字可能以多种格式出现：
    - 有/无千位分隔符: 1,280 vs 1280
    - 有/无小数点: 1280.0 vs 1280
    - 有/无货币符号: $1,280 vs 1,280
    - 括号表示负数: (1,280) vs -1,280
    - 不同小数位数: 1280.0 vs 1280.00

    Args:
        number_str: 原始数字字符串 (如 "1,280.0" 或 "-45.3%")

    Returns:
        去重后的变体列表，按优先级排序（精确匹配优先）
    """
    variants = set()
    cleaned = number_str.strip()

    # 加入原始形式
    variants.add(cleaned)

    # 移除货币符号和空格
    no_currency = re.sub(r'[$€£¥\s]', '', cleaned)
    variants.add(no_currency)

    # 移除百分号
    no_pct = no_currency.rstrip('%')
    if no_pct != no_currency:
        variants.add(no_pct)

    # 处理括号负数: (123) → -123
    paren_match = re.match(r'^\((.+)\)$', no_currency)
    if paren_match:
        inner = paren_match.group(1)
        variants.add(f"-{inner}")
        no_currency = inner  # 继续用内部值生成变体

    # 移除千位分隔符
    no_comma = no_currency.replace(',', '')
    variants.add(no_comma)

    # 有千位分隔符的版本
    variants.add(no_currency)

    # 处理小数: 有无小数点, 不同精度
    if '.' in no_comma:
        # 移除末尾的零和小数点
        no_trailing = no_comma.rstrip('0').rstrip('.')
        variants.add(no_trailing)
        # 加千位分隔符版本
        try:
            val = float(no_comma)
            variants.add(f"{val:,.0f}")
            variants.add(f"{val:,.1f}")
            variants.add(f"{val:,.2f}")
            variants.add(f"{val:.0f}")
            variants.add(f"{val:.1f}")
            variants.add(f"{val:.2f}")
        except ValueError:
            pass
    else:
        # 整数: 添加带小数点版本
        try:
            val = float(no_comma)
            variants.add(f"{val:.1f}")
            variants.add(f"{val:,.1f}")
        except ValueError:
            pass

    # 添加货币符号变体
    for v in list(variants):
        if not v.startswith('$') and not v.startswith('-'):
            variants.add(f"${v}")

    # 去重并按长度降序排列 (更长的匹配更精确)
    result = sorted(variants, key=lambda x: (-len(x), x))
    return result


def normalize_number(text: str) -> str | None:
    """
    将各种格式的数字文本归一化为标准浮点数字符串。

    Args:
        text: 数字文本, 如 "$1,280.50" 或 "(45.3%)"

    Returns:
        归一化后的字符串 (如 "1280.5"), 或 None 如果无法解析
    """
    cleaned = text.strip()

    # 移除前导符号
    cleaned = re.sub(r'^[$€£¥\s]+', '', cleaned)
    cleaned = cleaned.rstrip('%')

    # 处理括号负数
    paren_match = re.match(r'^\((.+)\)$', cleaned)
    if paren_match:
        cleaned = '-' + paren_match.group(1)

    # 移除千位分隔符
    cleaned = cleaned.replace(',', '')

    try:
        val = float(cleaned)
        # 去掉不必要的小数零
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        return None


def extract_numbers_from_text(text: str) -> list[str]:
    """
    从文本中提取所有数字 (含负数、小数、千位分隔符格式)。

    Args:
        text: 输入文本

    Returns:
        提取到的数字字符串列表 (原始格式)
    """
    # 匹配: 可选负号/括号 + 可选货币 + 数字(含逗号小数点)
    pattern = r'[\-\(]?\$?\d[\d,]*\.?\d*\)?%?'
    matches = re.findall(pattern, text)
    return [m for m in matches if any(c.isdigit() for c in m)]

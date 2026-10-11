"""
Tests for the pure catalog logic only - no MCP server, no AI model, no network. Same
separation week7/week8 use: the business rules (search, stock) are plain functions, so they're
tested directly instead of through an agent, which would be slow and non-deterministic.
"""

from catalog_mcp import catalog


def test_list_products_returns_every_product():
    products = catalog.list_products()
    assert len(products) == 8
    assert {p["id"] for p in products} == {f"p{i}" for i in range(1, 9)}


def test_get_product_finds_an_existing_product():
    product = catalog.get_product("p1")
    assert product is not None
    assert product["name"] == "Wireless Headphones"


def test_get_product_returns_none_for_an_unknown_id():
    assert catalog.get_product("does-not-exist") is None


def test_search_matches_name_category_or_description_case_insensitively():
    assert {p["id"] for p in catalog.search_products("headphones")} == {"p1"}
    assert {p["id"] for p in catalog.search_products("ELECTRONICS")} == {"p1", "p2", "p3", "p7"}
    assert {p["id"] for p in catalog.search_products("fast charging")} == {"p3"}  # matches description


def test_search_returns_nothing_for_a_blank_or_unmatched_query():
    assert catalog.search_products("") == []
    assert catalog.search_products("   ") == []
    assert catalog.search_products("spaceship") == []


def test_check_stock_reports_available_units():
    assert catalog.check_stock("p1") == "Wireless Headphones (p1) has 12 unit(s) in stock."


def test_check_stock_reports_out_of_stock_distinctly_from_available():
    # p2 has stock: 0 in data/products.json - must not be phrased like it has units available.
    message = catalog.check_stock("p2")
    assert "OUT OF STOCK" in message
    assert "has 0 unit" not in message


def test_check_stock_reports_an_unknown_product_id_instead_of_crashing():
    message = catalog.check_stock("does-not-exist")
    assert "No product found" in message

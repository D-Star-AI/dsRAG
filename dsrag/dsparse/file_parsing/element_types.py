from ..models.types import ElementType


def get_visual_elements_as_str(elements: list[ElementType]) -> str:
    visual_elements = [element["name"] for element in elements if element["is_visual"]]
    last_element = visual_elements[-1]
    if len(visual_elements) > 1:
        visual_elements[-1] = f"and {last_element}"
    return ", ".join(visual_elements)

def get_non_visual_elements_as_str(elements: list[ElementType]) -> str:
    non_visual_elements = [element["name"] for element in elements if not element["is_visual"]]
    last_element = non_visual_elements[-1]
    if len(non_visual_elements) > 1:
        non_visual_elements[-1] = f"and {last_element}"
    return ", ".join(non_visual_elements)

def get_num_visual_elements(elements: list[ElementType]) -> int:
    return len([element for element in elements if element["is_visual"]])

def get_num_non_visual_elements(elements: list[ElementType]) -> int:
    return len([element for element in elements if not element["is_visual"]])

def get_element_description_block(elements: list[ElementType]) -> str:
    element_blocks = []
    for element in elements:
        element_block = ELEMENT_PROMPT.format(
            element_name=element["name"],
            element_instructions=element["instructions"],
        )
        element_blocks.append(element_block)
    return "\n".join(element_blocks)

ELEMENT_PROMPT = """
- {element_name}
    - {element_instructions}
""".strip()

default_element_types = [
    {
        "name": "NarrativeText",
        "instructions": "This is the main text content of the page, including paragraphs, lists, titles, and any other text content that is not part of one of the other more specialized element types. Not all pages have narrative text, but most do. Be sure to use Markdown formatting for the text content. This includes using tags like # for headers, * for lists, etc. Make sure your header tags are properly nested and that your lists are properly formatted.",
        "is_visual": False,
    },
    {
        "name": "Figure",
        "instructions": "This covers charts, graphs, diagrams, etc. Associated titles, legends, axis titles, etc. should be considered to be part of the figure.",
        "is_visual": True,
    },
    {
        "name": "Image",
        "instructions": "This is any visual content on the page that doesn't fit into any of the other categories. This could include photos, illustrations, etc. Any title or captions associated with the image should be considered part of the image.",
        "is_visual": True,
    },
    {
        "name": "Table",
        "instructions": "This covers any tabular data arrangement on the page, including simple and complex tables. Any titles, captions, or notes associated with the table should be considered part of the table element.",
        "is_visual": True,
    },
    {
        "name": "Equation",
        "instructions": "This covers mathematical equations, formulas, and expressions that appear on the page. Associated equation numbers or labels should be considered part of the equation.",
        "is_visual": True,
    },
    {
        "name": "Header",
        "instructions": "This is the header of the page, which would be located at the very top of the page and may include things like page numbers. The text in a Header element is usually in a very small font size. This is NOT the same things as a Markdown header or heading. You should never user more than one header element per page. Not all pages have a header. Note that headers are not the same as titles or subtitles within the main text content of the page. Those should be included in NarrativeText elements.",
        "is_visual": False,
    },
    {
        "name": "Footnote",
        "instructions": "Footnotes should always be included as a separate element from the main text content as they aren't part of the main linear reading flow of the page. Not all pages have footnotes.",
        "is_visual": False,
    },
    {
        "name": "Footer",
        "instructions": "This is the footer of the page, which would be located at the very bottom of the page. You should never user more than one footer element per page. Not all pages have a footer, but when they do it is always the very last element on the page.",
        "is_visual": False,
    }
]


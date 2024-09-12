import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from dsrag.auto_context import get_document_title, get_document_summary, get_section_summary, get_chunk_header, get_segment_header
from dsrag.llm import OpenAIChatAPI

SECTION_TEXT = """
PRODUCT RESEARCH, DESIGN AND DEVELOPMENT
We believe our research, design and development efforts are key factors in our success. Technical innovation in the design and 
manufacturing process of footwear, apparel and athletic equipment receives continued emphasis as we strive to produce 
products that help to enhance athletic performance, reduce injury and maximize comfort, while decreasing our environmental 
impact.
In addition to our own staff of specialists in the areas of biomechanics, chemistry, exercise physiology, engineering, digital 
technologies, industrial design, sustainability and related fields, we also utilize research committees and advisory boards made 
up of athletes, coaches, trainers, equipment managers, orthopedists, podiatrists, physicians and other experts who consult with 
us and review certain designs, materials and concepts for product and manufacturing, design and other process improvements 
and compliance with product safety regulations around the world. E mployee athletes, athletes engaged under sports marketing 
contracts and other athletes wear-test and evaluate products during the design and development process.
As we continue to develop new technologies, we are simultaneously focused on the design of innovative products and 
experiences incorporating such technologies throughout our product categories and consumer applications. Using market 
intelligence and research, our various design teams identify opportunities to leverage new technologies in existing categories to 
respond to consumer preferences. The proliferation of Nike Air, Zoom, Free, Dri-FIT, Flyknit, FlyEase, ZoomX, Air Max, React and 
Forward technologies, among others, typifies our dedication to designing innovative products.
"""


class TestAutoContext(unittest.TestCase):
    def test__get_document_title(self):
        auto_context_model = OpenAIChatAPI(model="gpt-4o-mini")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "../data/nike_2023_annual_report.txt")
        with open(file_path, "r") as f:
            document_text = f.read()
        document_title_guidance = "Make sure the title is nice and human readable."
        document_title = get_document_title(auto_context_model, document_text, document_title_guidance=document_title_guidance)
        assert "nike" in document_title.lower()

    def test__get_document_summary(self):
        document_title = "NIKE, Inc. Annual Report on Form 10-K for the Fiscal Year Ended May 31, 2023"
        auto_context_model = OpenAIChatAPI(model="gpt-4o-mini")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "../data/nike_2023_annual_report.txt")
        with open(file_path, "r") as f:
            document_text = f.read()
        document_summarization_guidance = "Make sure the summary is concise and informative."
        document_summary = get_document_summary(auto_context_model, document_text, document_title, document_summarization_guidance=document_summarization_guidance)
        assert "nike" in document_summary.lower()

    def test__get_section_summary(self):
        document_title = "NIKE, Inc. Annual Report on Form 10-K for the Fiscal Year Ended May 31, 2023"
        section_title = "Product Research, Design and Development"
        auto_context_model = OpenAIChatAPI(model="gpt-4o-mini")
        section_text = SECTION_TEXT
        section_summarization_guidance = "Make sure the summary is concise and informative."
        section_summary = get_section_summary(auto_context_model, section_text, document_title, section_title, section_summarization_guidance=section_summarization_guidance)
        assert "product" in section_summary.lower()

    def test__get_chunk_header(self):
        document_title = "NIKE, Inc. Annual Report on Form 10-K for the Fiscal Year Ended May 31, 2023"
        document_summary = "This document is about: the financial performance, business operations, and strategic initiatives of NIKE, Inc. for the fiscal year ended May 31, 2023, as detailed in its Annual Report on Form 10-K."
        section_title = "Product Research, Design and Development"
        section_summary = "This section is about: Nike's commitment to product research, design, and development aimed at enhancing athletic performance and sustainability through technical innovation and expert collaboration."
        chunk_header = get_chunk_header(document_title, document_summary, section_title, section_summary)
        assert "Document context" in chunk_header
        assert "Section context" in chunk_header
        assert "Product Research, Design and Development" in chunk_header
        assert "NIKE, Inc. Annual Report on Form 10-K for the Fiscal Year Ended May 31, 2023" in chunk_header

    def test__get_segment_header(self):
        document_title = "NIKE, Inc. Annual Report on Form 10-K for the Fiscal Year Ended May 31, 2023"
        document_summary = "This document is about: the financial performance, business operations, and strategic initiatives of NIKE, Inc. for the fiscal year ended May 31, 2023, as detailed in its Annual Report on Form 10-K."
        segment_header = get_segment_header(document_title, document_summary)
        assert "Document context" in segment_header
        assert "NIKE, Inc. Annual Report on Form 10-K for the Fiscal Year Ended May 31, 2023" in segment_header
        assert "financial performance" in segment_header

class TestAutoContextNonEnglish(unittest.TestCase):
    def test__get_document_title(self):
        auto_context_model = OpenAIChatAPI(model="gpt-4o-mini")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "../data/les_miserables.txt")
        with open(file_path, "r") as f:
            document_text = f.read()
        document_title_guidance = ""
        document_title = get_document_title(auto_context_model, document_text, document_title_guidance=document_title_guidance, language="fr")
        assert "rables" in document_title.lower() # (part of Les Misérables)

    def test__get_document_summary(self):
        document_title = "Les Miserables"
        auto_context_model = OpenAIChatAPI(model="gpt-4o-mini")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "../data/les_miserables.txt")
        with open(file_path, "r") as f:
            document_text = f.read()
        document_summarization_guidance = ""
        document_summary = get_document_summary(auto_context_model, document_text, document_title, document_summarization_guidance=document_summarization_guidance, language="fr")
        assert "concerne" in document_summary.lower()

if __name__ == "__main__":
    unittest.main()
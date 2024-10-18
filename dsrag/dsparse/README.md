# dsParse
dsParse is a sub-module of dsRAG that does file parsing and chunking. You provide a file path and receive nice clean chunks.

```python
sections, chunks = parse_and_chunk_vlm(file_path)
```

## File parsing
dsParse uses a vision language model (VLM) to parse documents. 

The default model, `gemini-1.5-pro-002`, is the only model that works reliably enough for this task right now. It can be accessed through the Gemini API or the Vertex API.

#### Element types
Page content is categorized into the following seven categories:
- NarrativeText
    - This is the main text content of the page, including paragraphs, lists, titles, and any other text content that is not part of a header, footer, figure, table, or image.
- Figure
    - This covers charts, graphs, diagrams, etc. Associated titles, legends, axis titles, etc. are also included.
- Image
    - This is any visual content on the page that isn't a figure.
- Table
    - This is a table on the page. If the table can be represented accurately using Markdown, then it should be included as a Table element. If not, it should be included as an Image element to ensure accuracy.
- Header
    - This is the header of the page.
- Footnote
    - This is a footnote on the page. Footnotes should always be included as a separate element from the main text content as they aren't part of the main linear reading flow of the page.
- Footer
    - This is the footer of the page.

You can choose to exclude certain element types. By default, Header and Footer elements are excluded, as they rarely contain valuable information and they break up the flow between pages.

`exclude_elements = ["Header", "Footer", "Footnote"]

#### Bounding boxes for visual elements
For the two types of visual elements (Figure and Image) the VLM is required to include a bounding box for the image. This is what allows us to extract the image from the page.

## Semantic sectioning and chunking
Semantic sectioning uses an LLM to break a document into sections. It works by annotating the document with line numbers and then prompting an LLM to identify the starting lines for each “semantically cohesive section.” These sections should be anywhere from a few paragraphs to a few pages long. The sections then get broken into smaller chunks if needed. The LLM also generates descriptive titles for each section. These section titles get used in the contextual chunk headers created by AutoContext, which provides additional context to the ranking models (embeddings and reranker), enabling better retrieval.

The default model for semantic sectioning is `gpt-4o-mini`, but similarly strong models like `gemini-1.5-flash-002` will also work well. Less powerful models, like Claude 3 Haiku, don't work as well.

## Cost and latency/throughput estimation
An obvious concern with using a large model like `gemini-1.5-pro-002` to parse documents is the cost. Let's run the numbers:

VLM file parsing cost calculation (`gemini-1.5-pro-002`)
- image input: 1 image x $0.00032875 per image = $0.00032875
- text input (prompt): 400 tokens x $1.25/10^6 per token = $0.000500
- text output: 600 tokens x $5.00/10^6 per token = $0.003000
Total: $0.00382875/page or **$3.83 per 1000 pages**

This is actually cheaper than many commercially available PDF parsing services. Unstructured, for example, costs $10 per 1000 pages.

What about latency and throughput? Since each page is processed independently, this is a highly parallelizable problem. The main limiting factor then is the rate limits imposed by the VLM provider. The current rate limit for `gemini-1.5-pro-002` is 1000 requests per minute. Since dsParse uses one request per page, and processing a single page takes less than a minute, that means the limit is 1000 pages per minute. Processing a single page takes around 15-20 seconds, so that's the minimum latency for processing a document.

Semantic sectioning uses a much cheaper model, and it also uses far fewer output tokens, so it ends up being far cheaper than the file parsing step.

Semantic sectioning cost calculation (`gpt-4o-mini`)
- input: 800 tokens x $0.15/10^6 per token = $0.00012
- output: 50 tokens x $0.60/10^6 per token = $0.00003
Total: $0.00015/page or **$0.15 per 1000 pages**

Document text is processed in ~5000 token mega-chunks, which is roughly ten pages on average. But these mega-chunks have to be processed sequentially for each document. Processing each mega-chunk only takes a couple seconds, though, so even a large document of a few hundred pages will only take 20-60 seconds. Rate limits for the OpenAI API are heavily dependent on the usage tier you're in.
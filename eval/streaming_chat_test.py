import os
import sys
import time
from pathlib import Path

# Add parent directory to path to import dsrag
sys.path.append(str(Path(__file__).parent.parent))

from dsrag.chat import create_new_chat_thread, get_chat_thread_response
from dsrag.chat.chat_types import ChatThreadParams, ChatResponseInput
from dsrag.database.chat_thread.basic_db import BasicChatThreadDB
from dsrag.knowledge_base import KnowledgeBase
from rich.console import Console

def streaming_chat_test(test_streaming=True):
    """
    Test script for streaming chat functionality.
    This script will:
    1. Create a knowledge base with sample data
    2. Create a chat thread
    3. Send a message to the chat thread
    4. If test_streaming=True, also test streaming functionality
    
    Args:
        test_streaming (bool): Whether to test streaming functionality (defaults to True)
    """
    console = Console()
    console.print("[bold green]Starting Streaming Chat Test[/bold green]")
    
    # Check for API keys
    required_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
    for key in required_keys:
        if key not in os.environ:
            console.print(f"[bold red]Error: {key} not found in environment variables[/bold red]")
            return

    
    chat_db = BasicChatThreadDB()
    
    # Initialize knowledge base with test data
    console.print("Initializing knowledge base with test data...")
    test_data_path = Path(__file__).parent / "../tests/data/levels_of_agi.pdf"
    
    if not test_data_path.exists():
        console.print(f"[bold red]Error: Test data not found at {test_data_path}[/bold red]")
        return
    
    kb = KnowledgeBase(
        kb_id="test_kb",
        description="Test knowledge base containing AGI information",
    )
    kb.add_document(
        doc_id="levels_of_agi",
        file_path=str(test_data_path)
    )
    console.print("[green]Knowledge base created successfully[/green]")
    
    # Create chat thread
    console.print("Creating chat thread...")
    chat_params = ChatThreadParams(
        kb_ids=["test_kb"],
        model="claude-3-5-sonnet-20241022",
        temperature=0.2,
        system_message="You are a helpful assistant specialized in AI concepts.",
        target_output_length="medium"
    )
    
    thread_id = create_new_chat_thread(chat_params, chat_db)
    console.print(f"[green]Chat thread created with ID: {thread_id}[/green]")
    
    # Choose what to test based on the test_streaming flag
    user_query = "What are the different levels of AGI described in the document?"
    response_input = ChatResponseInput(
        user_input=user_query,
    )
    
    if test_streaming:
        console.print("\n[bold]Testing streaming response:[/bold]")
        try:
            # Reset static variables for fresh streaming display
            if hasattr(streaming_chat_test, 'header_printed'):
                del streaming_chat_test.header_printed
            if hasattr(streaming_chat_test, 'prev_response'):
                del streaming_chat_test.prev_response
                
            start_time = time.time()
            
            console.print("Sending message to chat thread with streaming enabled...")
            
            # Create a variable to accumulate the response for display
            full_response = ""
            
            # Process the streaming response
            for partial_response in get_chat_thread_response(
                thread_id=thread_id,
                get_response_input=response_input,
                chat_thread_db=chat_db,
                knowledge_bases={"test_kb": kb},
                stream=True
            ):
                # Try to get the current content safely
                try:
                    if isinstance(partial_response, dict) and "model_response" in partial_response:
                        current_content = partial_response["model_response"].get("content", "")
                    else:
                        # Fallback for unexpected response format
                        current_content = str(partial_response)
                    
                    # Make sure we have valid content (handle None case)
                    if current_content is None:
                        current_content = ""
                    
                    # Update the full response only if it's not None
                    full_response = current_content
                    
                    # Use a simpler approach that works better across different terminals
                    # First time, print the query
                    if not hasattr(streaming_chat_test, 'header_printed'):
                        console.print("[bold]Query:[/bold]", user_query)
                        console.print("\n[bold]Streaming response:[/bold]")
                        streaming_chat_test.header_printed = True
                    
                    # Instead of clearing the console, just print the new content
                    # Get the new content by comparing with previous response
                    if hasattr(streaming_chat_test, 'prev_response'):
                        prev_len = len(streaming_chat_test.prev_response) if streaming_chat_test.prev_response else 0
                        # Find what's new in this chunk, only if there is content
                        if full_response and len(full_response) > prev_len:
                            new_content = full_response[prev_len:]
                            # Print only the new part
                            print(new_content, end="", flush=True)
                    else:
                        # First chunk, print it all if it has content
                        if full_response:
                            print(full_response, end="", flush=True)
                    
                    # Save for next comparison (only if not None)
                    streaming_chat_test.prev_response = full_response
                except Exception as e:
                    console.print(f"[red]Error displaying partial response: {str(e)}[/red]")
                    console.print(f"[red]Partial response: {partial_response}[/red]")
                
                # No artificial delay needed for real streaming
            
            elapsed = time.time() - start_time
            
            # Print final response time
            console.print(f"\n[green]Streaming response completed in {elapsed:.2f} seconds[/green]")
            
            # Print citation information if available
            if "citations" in partial_response["model_response"] and partial_response["model_response"]["citations"]:
                console.print("\n[bold]Citations:[/bold]")
                for citation in partial_response["model_response"]["citations"]:
                    console.print(f"- {citation['doc_id']}: {citation['cited_text'][:100]}...")
        
        except Exception as e:
            console.print(f"[bold red]Streaming failed: {str(e)}[/bold red]")
            import traceback
            console.print(traceback.format_exc())
    
    else:
        console.print("\n[bold]Testing non-streaming response:[/bold]")
        start_time = time.time()
        
        console.print("Sending message to chat thread...")
        response = get_chat_thread_response(
            thread_id=thread_id,
            get_response_input=response_input,
            chat_thread_db=chat_db,
            knowledge_bases={"test_kb": kb}
        )
        
        elapsed = time.time() - start_time
        console.print(f"[green]Response received in {elapsed:.2f} seconds[/green]")
        console.print(response["model_response"]["content"])
        
        # Print citation information if available
        if "citations" in response["model_response"] and response["model_response"]["citations"]:
            console.print("\n[bold]Citations:[/bold]")
            for citation in response["model_response"]["citations"]:
                console.print(f"- {citation['doc_id']}: {citation['cited_text'][:100]}...")
    
    # Test complete
    console.print("[green]Test complete[/green]")

if __name__ == "__main__":
    # Set this to True to enable the streaming test
    TEST_STREAMING = True
    
    streaming_chat_test(test_streaming=TEST_STREAMING)
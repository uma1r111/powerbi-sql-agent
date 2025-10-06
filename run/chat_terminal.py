import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow.graph import agent


def main():
    print("=" * 60)
    print("PowerBI SQL Agent - Terminal Chat")
    print("=" * 60)
    print("Type your questions about the Northwind database.")
    print("Type 'exit' or 'quit' to end the session.\n")
    
    session_id = "terminal_session_001"
    
    while True:
        try:
            # Get user input
            user_query = input("\nYou: ").strip()
            
            if not user_query:
                continue
                
            if user_query.lower() in ['exit', 'quit', 'bye']:
                print("\nGoodbye! Session ended.")
                break
            
            # Process query
            print("\nAgent: Processing...\n")
            result = agent.process_query_sync(user_query, session_id=session_id)
            
            # Get the AI response from conversation history
            if result.messages and len(result.messages) > 0:
                last_message = result.messages[-1]
                print(f"Agent: {last_message.content}\n")
            else:
                print("Agent: [No response generated]\n")
            
            # Show debug info
            print(f"\n[Debug: Tables: {result.selected_tables}, Success: {result.processing_complete}]")
            
        except KeyboardInterrupt:
            print("\n\nSession interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try another query.\n")

if __name__ == "__main__":
    main()
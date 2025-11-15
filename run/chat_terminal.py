import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow.graph import agent


def print_banner():
    """Print welcome banner"""
    print("=" * 70)
    print("🤖 PowerBI SQL Agent - Conversational Interface")
    print("=" * 70)
    print("\n📊 Ask questions about the Northwind database in natural language")
    print("\n💡 Special Commands:")
    print("  /detailed     - Switch to detailed table view")
    print("  /simple       - Switch to conversational view (default)")
    print("  /sql on       - Show SQL queries in responses")
    print("  /sql off      - Hide SQL queries")
    print("  /debug on     - Show execution details")
    print("  /debug off    - Hide execution details")
    print("  /help         - Show this help")
    print("  /clear        - Clear conversation history")
    print("  exit/quit     - End session")
    print("=" * 70)
    print()

def print_help():
    """Print help information"""
    print("\n" + "=" * 70)
    print("📚 How to use:")
    print("=" * 70)
    print("\n🔹 Ask natural language questions:")
    print("  • How many customers do we have?")
    print("  • Show me customers from Germany")
    print("  • What's the total revenue by product category?")
    print("\n🔹 Follow-up questions:")
    print("  • How many orders did they place?")
    print("  • Show me their contact information")
    print("\n🔹 Toggle display modes:")
    print("  • /detailed - See full data tables")
    print("  • /simple - Get conversational summaries")
    print("  • /sql on - See the generated SQL")
    print("=" * 70)
    print()

def main():
    print_banner()
    
    session_id = "terminal_session_001"
    
    # Session state
    response_format = "conversational"
    show_sql = False
    show_debug = False
    
    # Track conversation for context
    conversation_count = 0
    
    while True:
        try:
            # Get user input
            user_input = input("\n💬 You: ").strip()
            
            if not user_input:
                continue
            
            # Handle exit commands
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\n👋 Goodbye! Session ended.")
                print(f"📊 Total queries processed: {conversation_count}")
                break
            
            # Handle special commands
            if user_input.startswith('/'):
                command = user_input.lower()
                
                if command == '/help':
                    print_help()
                    continue
                
                elif command == '/detailed':
                    response_format = "detailed"
                    print("✅ Switched to detailed table view")
                    continue
                
                elif command == '/simple':
                    response_format = "conversational"
                    print("✅ Switched to conversational view")
                    continue
                
                elif command == '/sql on':
                    show_sql = True
                    print("✅ SQL queries will be shown in responses")
                    continue
                
                elif command == '/sql off':
                    show_sql = False
                    print("✅ SQL queries will be hidden")
                    continue
                
                elif command == '/debug on':
                    show_debug = True
                    print("✅ Debug information enabled")
                    continue
                
                elif command == '/debug off':
                    show_debug = False
                    print("✅ Debug information disabled")
                    continue
                
                elif command == '/clear':
                    session_id = f"terminal_session_{conversation_count + 1}"
                    conversation_count = 0
                    print("✅ Conversation history cleared. Starting fresh session.")
                    continue
                
                else:
                    print(f"❌ Unknown command: {user_input}")
                    print("💡 Type /help to see available commands")
                    continue
            
            # Process query with current settings
            print("\n🔄 Processing", end="", flush=True)
            for _ in range(3):
                print(".", end="", flush=True)
                import time
                time.sleep(0.3)
            print()
            
            # Create initial state with formatting preferences
            from state.agent_state import StateManager, AgentState
            initial_state = StateManager.create_initial_state(user_input)
            initial_state.session_id = session_id
            initial_state.response_format = response_format
            initial_state.show_sql = show_sql
            initial_state.show_execution_details = show_debug
            
            # Process through agent
            result = agent.process_query_sync(user_input, session_id=session_id)
            
            # Update result with current preferences (in case state wasn't passed correctly)
            result.response_format = response_format
            result.show_sql = show_sql
            result.show_execution_details = show_debug
            
            # Get the AI response
            if result.messages and len(result.messages) > 0:
                last_message = result.messages[-1]
                print(f"\n🤖 Agent:\n{last_message.content}\n")
            else:
                print("\n🤖 Agent: [No response generated]\n")
            
            # Show debug info if enabled
            if show_debug:
                print(f"\n🔍 Debug Info:")
                print(f"   Tables: {result.selected_tables}")
                print(f"   Success: {result.processing_complete}")
                print(f"   Intent: {result.business_intent}")
                print(f"   Complexity: {result.query_complexity}")
                if result.errors:
                    print(f"   Errors: {result.errors}")
            
            conversation_count += 1
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Session interrupted.")
            print(f"📊 Queries processed: {conversation_count}")
            print("👋 Goodbye!")
            break
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("💡 Please try another query or type /help for assistance.\n")
            import traceback
            if show_debug:
                print("\n🐛 Full error trace:")
                traceback.print_exc()

if __name__ == "__main__":
    main()
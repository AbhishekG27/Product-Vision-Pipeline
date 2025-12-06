"""
Simple script to start the UI with proper Windows configuration.
"""

import subprocess
import sys
import webbrowser
import time

print("=" * 60)
print("Starting Product Vision Pipeline UI...")
print("=" * 60)
print()

# Try to start the app
try:
    # Import and run
    from app import demo
    
    print("Opening browser at http://localhost:7860")
    print("Press Ctrl+C to stop the server")
    print()
    
    # Launch with localhost
    demo.launch(
        share=False,
        server_name="127.0.0.1",  # Use localhost IP
        server_port=7860,
        inbrowser=True,  # Auto-open browser
        show_error=True
    )
except KeyboardInterrupt:
    print("\n\nServer stopped by user")
except Exception as e:
    print(f"\nError starting server: {e}")
    print("\nTrying alternative method...")
    
    # Alternative: use subprocess
    try:
        subprocess.run([sys.executable, "app.py"])
    except Exception as e2:
        print(f"Error: {e2}")
        print("\nPlease run manually: python app.py")


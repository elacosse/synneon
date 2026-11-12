import platform
import subprocess
import sys


class TTS:
    def __init__(self, rate: int = 200, volume: float = 1.0):
        """
        Initialize the TTS engine.

        Args:
            rate: approximate words per minute. Default 200 is reasonable for 'say'.
            volume: currently unused for macOS 'say' command simple wrapper.
        """
        self.rate = rate
        self.system = platform.system()

    def _speak_macos(self, text: str):
        """Use the macOS 'say' command."""
        try:
            # -r is rate in words per minute
            subprocess.run(["say", "-r", str(self.rate), text], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error speaking text: {e}", file=sys.stderr)
        except FileNotFoundError:
            print("Error: 'say' command not found. Are you on macOS?", file=sys.stderr)

    def speak_text(self, text: str):
        """Speaks the text as a whole sentence/phrase."""
        if self.system == "Darwin":
            self._speak_macos(text)
        else:
            print(f"[TTS Simulator] ({self.system}): {text}")

    def speak_letters(self, text: str):
        """Speaks each character of the string individually."""
        for char in text:
            if char.strip():
                if self.system == "Darwin":
                    # Speak the character.
                    # For special characters or clarity, sometimes mapping is needed,
                    # but 'say' handles most chars well.
                    self._speak_macos(char)
                else:
                    print(f"[TTS Simulator] char: {char}")


def main():
    tts = TTS()
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        print(f"Speaking letters of: {text}")
        tts.speak_letters(text)
    else:
        print("Usage: python -m synneon.input.tts <text>")
        print("Example: python -m synneon.input.tts hello")
        tts.speak_letters("hello")


if __name__ == "__main__":
    main()

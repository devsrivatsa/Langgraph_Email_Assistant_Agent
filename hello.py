from jinja2 import Template
from pathlib import Path

def main():
    print("Hello from email-assistant!")


if __name__ == "__main__":
    _ROOT = Path(__file__).parent.absolute()
    _TEMPLATE = Path(_ROOT, "test.jinja2")

    print(Template(open(_TEMPLATE)).render(name="Srivatsa"))

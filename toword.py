import pypandoc
import argparse

def md_to_docx(input_md, output_docx):
    """
    Convert a Markdown file to a Word document.

    Args:
        input_md (str): Path to the input Markdown file.
        output_docx (str): Path for the output Word document.
    """
    try:
        pypandoc.convert_file(input_md, 'docx', outputfile=output_docx)
        print(f"Successfully converted '{input_md}' to '{output_docx}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    #parser = argparse.ArgumentParser(description='Convert Markdown to Word document.')
    #parser.add_argument('input_md', help='Path to the input Markdown file.')
    #parser.add_argument('output_docx', help='Path for the output Word document.')
    #args = parser.parse_args()

    #files = ['./audio/mom.md', './audio/transcript.md', './audio/report.md']
    files = ['./audio/transcript.md', './audio/report2.md']

    for file in files:
        md_to_docx(file, file.replace('.md', '.docx')) 

if __name__ == "__main__":
    main()

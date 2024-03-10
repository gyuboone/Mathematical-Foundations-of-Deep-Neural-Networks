from fpdf import FPDF

def convert_to_pdf(input_file, output_file):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    with open(input_file, "r", encoding="utf-8") as file:
        for line in file:
            line = line.rstrip()
            pdf.multi_cell(200, 5, txt=line)
    
    pdf.output(output_file)

if __name__ == "__main__":
    input_file = "../assignment1/problem 5/"+"conv1D.py"  # 변환할 Python 파일
    output_file = "../assignment1/problem 5/"+"conv1D_multi.pdf"  # 생성될 PDF 파일
    convert_to_pdf(input_file, output_file)
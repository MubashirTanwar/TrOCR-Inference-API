from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import os
from .prepo import extract_pages_from_pdf
from .prepo import process_image
from .models import PDFDocument, ExtractedLine
from .test import preview
from .wordbox import get_string
@api_view(['POST'])
def extract_lines(request):
    if request.method == 'POST':
        pdf_file = request.data['file']
        temp_dir = r'pdf_api\temp_images'
        output_dir = r'pdf_api\output_images'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        uploaded_pdf = PDFDocument(file=pdf_file)
        uploaded_pdf.save()
        pdf_path = uploaded_pdf.file.path 
        page_images = extract_pages_from_pdf(pdf_path, temp_dir)
        
        text = ""
        for page_num, page_image in enumerate(page_images):
            num_lines = 0
            total_lines = get_string(page_image, output_dir, page_num)
            for num_lines, filename in enumerate(sorted(os.listdir(output_dir)), start=1):
                img_path = os.path.join(output_dir, filename)
                text += " " + preview(img_path)
                total_lines -= 1
                print(text)
                extracted_lines = ExtractedLine(pdf_document=uploaded_pdf, image=img_path, text=f"{num_lines} lines")
                extracted_lines.save()
            return Response({'num_lines': total_lines, 'text' : text}, status=status.HTTP_200_OK)
        return Response(status=status.HTTP_405_METHOD_NOT_ALLOWED)
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import os
from .prepo import extract_pages_from_pdf
from .prepo import process_image

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
        pdf_path = os.path.join(temp_dir, 'uploaded_pdf.pdf')
        with open(pdf_path, 'wb') as f:
            f.write(pdf_file.read())
        page_images = extract_pages_from_pdf(pdf_path, temp_dir)
        num_lines = 0
        for page_num, page_image in enumerate(page_images):
            process_image(page_image, output_dir, page_num)
            num_lines += len([name for name in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, name))])
        return Response({'num_lines': num_lines}, status=status.HTTP_200_OK)
    return Response(status=status.HTTP_405_METHOD_NOT_ALLOWED)
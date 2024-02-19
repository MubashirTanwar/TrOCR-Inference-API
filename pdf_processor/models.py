from django.db import models

class PDFDocument(models.Model):
    file = models.FileField(upload_to='pdf_files/')

class ExtractedLine(models.Model):
    pdf_document = models.ForeignKey(PDFDocument, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='extracted_images/')

import numpy as np
import pandas as pd


from tabulate import tabulate


from reportlab.pdfgen import canvas # pyright: ignore[reportMissingModuleSource]

teste = pd.read_csv("../resultados/default/resumo_resultado_default_att.csv", sep = ";")

#         #     1   0
#         # 1  TP  FN
#         # 0  FP  TN

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle

c = canvas.Canvas("../resultados/default/confusion_matrix.pdf", pagesize=letter)
c.setFont("Helvetica", 12)

posicao_y = 750

for i in range(len(teste)):
    if posicao_y < 100:
        c.showPage()
        posicao_y = 750

    nome_arquivo = teste['arquivo'][i].replace('.csv', '')
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, posicao_y, f"Setup: {nome_arquivo}")
    posicao_y -= 25
    matriz = [
        ["", "Tuning", "Default"],
        ["Tuning", str(teste["TP"][i]), str(teste["FN"][i])],
        ["Default", str(teste["FP"][i]), str(teste["TN"][i])]
    ]

    tabela = Table(matriz, colWidths=[80, 80, 80])

    tabela.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))

    tabela.wrapOn(c, 400, 200)
    tabela.drawOn(c, 100, posicao_y - 60)

    posicao_y -= 100

c.save()
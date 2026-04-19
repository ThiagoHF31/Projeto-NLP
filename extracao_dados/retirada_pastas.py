import shutil
import os
from pathlib import Path

origem = Path("C:/Users/paogr/Desktop/NLP/dados/pdfs")
destino = Path("C:/Users/paogr/Desktop/NLP/dados/pdfs")

def destino_unico(caminho):
    if not caminho.exists():
        return caminho
    i = 1
    while Path(f"{caminho}_{i}").exists():
        i += 1
    return Path(f"{caminho}_{i}")

try:
    while True:
        pastas = [p for p in origem.iterdir() if p.is_dir()]
        if not pastas:
            break

        for cada in pastas:
            for cada_int in cada.iterdir():
                caminho_base = cada / cada_int.name
                caminho_final = destino_unico(destino / cada_int.name)
                shutil.move(str(caminho_base), str(caminho_final))
                print(f"Pasta {cada_int.name} deslocada para {caminho_final}")
            shutil.rmtree(cada)

except KeyboardInterrupt:
    print("Interrompido! Pastas restantes não foram processadas.")





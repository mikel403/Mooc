from Mikel_subtitler2_adj_v4 import generate_vtt_from_audio
import spacy
import time
import os
import torch, gc

start = time.perf_counter()

# generate_vtt_from_audio("ModeloPPTXUNED-SEFH (1).mp4", "ModeloPPTXUNED-SEFH (1).vtt")

# Rutas
carpeta_entrada = "videos_subtitular"
carpeta_salida = "subtitulos_generados"

gc.collect()
torch.cuda.empty_cache()
# Procesar todos los .mp4
for archivo in os.listdir(carpeta_entrada):
    if archivo.endswith(".mp4"):
        ruta_video = os.path.join(carpeta_entrada, archivo)
        nombre_base = os.path.splitext(archivo)[0]
        salida_vtt = os.path.join(carpeta_salida, f"{nombre_base}.vtt")
        
        print(f"⏳ Procesando {archivo}...")
        generate_vtt_from_audio(ruta_video, salida_vtt)
        print(f"✅ Subtítulo generado: {salida_vtt}")
        gc.collect()
        torch.cuda.empty_cache()


end = time.perf_counter()
print(f"Tiempo: {end - start:.4f} segundos")

# nlp = spacy.load("es_core_news_lg")
# line="Afecta a la salud y a la calidad de vida en diversos ámbitos"
# doc = nlp(line.strip())

# for token in doc:
#     print(token)

#     print(token.dep_)
#     print(token.pos_)

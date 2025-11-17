"""
MAIN – Ejecución del sistema de 3 agentes:
RAG → SQL → JUEZ → Respuesta final sintetizada
"""

import sys
from judge import judge_agents    # el archivo donde está tu juez optimizado

def run_interactive():
    print("\n==============================================")
    print("   SISTEMA DE AGENTES CLÍNICOS – MODO CLI")
    print("==============================================\n")

    # Si no hay argumentos → pedir input interactivo
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Ingresa tu consulta médica: ")

    print(f"\n→ Ejecutando agentes para la consulta:\n   \"{query}\"\n")

    result = judge_agents(query)

    # -------------------------------
    # SALIDA FORMATEADA
    # -------------------------------
    print("\n================= RESULTADOS =================\n")

    print("----- RESPUESTA RAG -----\n")
    print(result["rag_response"])
    print("\n----------------------------------------------\n")

    print("----- RESPUESTA SQL -----\n")
    print(result["sql_response"])
    print("\n----------------------------------------------\n")

    print("----- JUICIO DEL AGENTE JUEZ -----\n")
    print(f"Ganador: {result['judgment']['winner']}")
    print(f"Score RAG: {result['judgment']['rag_score']}")
    print(f"Score SQL: {result['judgment']['sql_score']}")
    print(f"Razón: {result['judgment']['reason']}")
    print("\n----------------------------------------------\n")

    print("========== RESPUESTA FINAL SINTETIZADA ==========\n")
    print(result["final_answer"])
    print("\n==================================================\n")


if __name__ == "__main__":
    run_interactive()

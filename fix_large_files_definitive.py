#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔥 SOLUTION DÉFINITIVE: BFG REPO-CLEANER
Supprime définitivement les gros fichiers de TOUT l'historique Git
"""

import subprocess
import sys
import os
import requests
from pathlib import Path

def download_bfg():
    """Télécharge BFG Repo-Cleaner"""
    print("📥 Téléchargement de BFG Repo-Cleaner...")
    
    bfg_url = "https://repo1.maven.org/maven2/com/madgag/bfg/1.14.0/bfg-1.14.0.jar"
    bfg_path = Path("bfg.jar")
    
    if bfg_path.exists():
        print("✅ BFG déjà téléchargé")
        return str(bfg_path)
    
    try:
        response = requests.get(bfg_url, stream=True)
        response.raise_for_status()
        
        with open(bfg_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✅ BFG téléchargé: {bfg_path}")
        return str(bfg_path)
    
    except Exception as e:
        print(f"❌ Erreur téléchargement BFG: {e}")
        return None

def run_command(command, capture_output=True):
    """Exécute une commande et retourne le résultat"""
    try:
        print(f"⚙️ Exécution: {command}")
        result = subprocess.run(command, shell=True, capture_output=capture_output, text=True)
        if result.returncode == 0:
            print("   ✅ Succès")
        else:
            print(f"   ⚠️ Code retour: {result.returncode}")
            if result.stderr:
                print(f"   Erreur: {result.stderr}")
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False, "", str(e)

def clean_with_bfg():
    """Utilise BFG pour nettoyer l'historique"""
    print("🔥 NETTOYAGE AVEC BFG REPO-CLEANER")
    print("=" * 50)
    
    # Télécharger BFG
    bfg_path = download_bfg()
    if not bfg_path:
        print("❌ Impossible de télécharger BFG")
        return False
    
    # Vérifier Java
    success, _, _ = run_command("java -version")
    if not success:
        print("❌ Java requis pour BFG. Installez Java d'abord.")
        return False
    
    print("\n🧹 Nettoyage de l'historique...")
    
    # Sauvegarde
    print("💾 Création d'une sauvegarde...")
    run_command("git branch backup-avant-bfg")
    
    # Nettoyer les gros fichiers avec BFG
    commands = [
        f"java -jar {bfg_path} --delete-files '*.zip' .",
        f"java -jar {bfg_path} --delete-files '*.csv' .",
        f"java -jar {bfg_path} --delete-files '*.safetensors' .",
        f"java -jar {bfg_path} --delete-files '*.bin' .",
        f"java -jar {bfg_path} --delete-folders 'data' .",
        f"java -jar {bfg_path} --delete-folders 'colab_package' .",
        f"java -jar {bfg_path} --strip-blobs-bigger-than 50M ."
    ]
    
    for cmd in commands:
        success, stdout, stderr = run_command(cmd)
        if success:
            print(f"   ✅ Nettoyage réussi")
        else:
            print(f"   ⚠️ Avertissement: {stderr}")
    
    # Finaliser le nettoyage
    print("\n🔧 Finalisation...")
    cleanup_commands = [
        "git reflog expire --expire=now --all",
        "git gc --prune=now --aggressive"
    ]
    
    for cmd in cleanup_commands:
        run_command(cmd, capture_output=False)
    
    print("\n✅ Nettoyage BFG terminé!")
    return True

def manual_solution():
    """Solution manuelle sans BFG"""
    print("🛠️ SOLUTION MANUELLE ALTERNATIVE")
    print("=" * 50)
    
    print("Si BFG ne fonctionne pas, voici la solution manuelle:")
    print()
    print("1️⃣ CRÉER UN NOUVEAU DÉPÔT PROPRE")
    print("   git checkout --orphan nouvelle-branche")
    print("   git rm -rf .")
    print()
    print("2️⃣ COPIER SEULEMENT LES FICHIERS ESSENTIELS")
    essential_files = [
        "app.py", "start.py", "config.py", ".gitignore",
        "static/", "templates/", "utils/", "tests/", "docs/", "scripts/",
        "README*.md", "requirements.txt"
    ]
    print("   Fichiers à conserver:")
    for f in essential_files:
        print(f"   • {f}")
    
    print()
    print("3️⃣ NOUVEAU COMMIT PROPRE")
    print("   git add .")
    print("   git commit -m 'Clean start: Remove all large files'")
    print("   git branch -D main")
    print("   git branch -m main")
    print("   git push origin main --force")

def quick_fix():
    """Solution rapide : nouveau dépôt"""
    print("⚡ SOLUTION RAPIDE: NOUVEAU DÉPÔT")
    print("=" * 50)
    
    print("La solution la plus simple :")
    print()
    print("1. Créer un NOUVEAU dépôt sur GitHub (ex: NLP-Sentiment-Analysis-Clean)")
    print("2. Changer l'origine :")
    print("   git remote set-url origin https://github.com/USERNAME/NLP-Sentiment-Analysis-Clean.git")
    print("3. Push vers le nouveau dépôt :")
    print("   git push origin main")
    print()
    print("✅ Avantages: Simple, rapide, pas de risque")
    print("❌ Inconvénient: Perd l'historique des commits")

def main():
    """Menu principal"""
    print("🔥 SOLUTION DÉFINITIVE POUR LES GROS FICHIERS")
    print("=" * 60)
    print("GitHub refuse encore le push à cause des gros fichiers dans l'historique.")
    print()
    print("Choisissez une solution :")
    print("1. 🔥 BFG Repo-Cleaner (Recommandé)")
    print("2. 🛠️ Solution manuelle")
    print("3. ⚡ Nouveau dépôt (Simple)")
    print("4. ❌ Annuler")
    print()
    
    choice = input("Votre choix (1-4): ").strip()
    
    if choice == "1":
        if clean_with_bfg():
            print("\n🚀 PROCHAINE ÉTAPE:")
            print("   git push origin main --force")
        else:
            print("\n❌ BFG a échoué. Essayez l'option 2 ou 3.")
    
    elif choice == "2":
        manual_solution()
    
    elif choice == "3":
        quick_fix()
    
    elif choice == "4":
        print("❌ Opération annulée")
    
    else:
        print("❌ Choix invalide")

if __name__ == "__main__":
    main()

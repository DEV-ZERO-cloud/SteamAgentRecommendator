"""
prepare_data.py – Prepara el dataset de juegos RPG de Steam.

Opciones:
  1. Descarga automática desde Kaggle (requiere kaggle CLI configurado)
  2. Genera un dataset sintético de 500 juegos RPG para pruebas

Uso:
    python prepare_data.py --mode synthetic    # genera datos de prueba
    python prepare_data.py --mode kaggle       # descarga dataset real
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).parent / "src" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = DATA_DIR / "steam_rpg_games.csv"

# ── Datos sintéticos ──────────────────────────────────────────────────────────
RPG_NAMES = [
    "Elden Ring", "The Witcher 3: Wild Hunt", "Baldur's Gate 3",
    "Dark Souls III", "Pathfinder: Wrath of the Righteous",
    "Divinity: Original Sin 2", "Dragon Age: Origins", "Pillars of Eternity",
    "Disco Elysium", "Tyranny", "Planescape: Torment", "Neverwinter Nights 2",
    "Gothic", "Arcanum", "Fallout 2", "Wasteland 3",
    "Shadowrun: Dragonfall", "Vampire: The Masquerade", "Torchlight II",
    "Grim Dawn", "Path of Exile", "Diablo IV", "Hades", "Dead Cells",
    "Monster Hunter: World", "Final Fantasy XIV", "Dragon's Dogma 2",
    "Tales of Arise", "Persona 5 Royal", "Yakuza: Like a Dragon",
    "NieR: Automata", "Star Wars: KOTOR", "Mass Effect Legendary Edition",
    "Kingdom Come: Deliverance", "Mount & Blade II: Bannerlord",
    "Morrowind", "Oblivion", "Skyrim", "Enderal: Forgotten Stories",
    "Dragon's Dogma: Dark Arisen", "Two Worlds II", "Gothic 3",
    "Sacred 2", "Titan Quest Anniversary Edition", "Victor Vran",
    "Wolcen: Lords of Mayhem", "Warhammer: Chaosbane", "Last Epoch",
    "Poe 2: Deadfire", "Solasta: Crown of the Magister",
]

TAG_POOLS = {
    "action_rpg":   ["Action RPG","Hack and Slash","Action","Combat","Loot","Third Person"],
    "jrpg":         ["JRPG","Anime","Turn-Based","Story Rich","Visual Novel","Linear"],
    "open_world":   ["Open World","Exploration","Sandbox","Survival","Crafting","Non-linear"],
    "tactical_rpg": ["Tactical RPG","Strategy","Turn-Based Strategy","Party-Based","Isometric"],
    "roguelike":    ["Roguelike","Roguelite","Procedural Generation","Dungeon Crawler","Perma-death"],
    "crpg":         ["RPG","Party-Based RPG","Classic RPG","Isometric","Narrative","Choices Matter"],
    "soulslike":    ["Souls-like","Difficult","Dark Fantasy","Action","Boss Rush","Atmospheric"],
}

DEVELOPERS = [
    "FromSoftware","CD Projekt Red","Larian Studios","Obsidian Entertainment",
    "inXile Entertainment","BioWare","Piranha Bytes","Troika Games",
    "Grinding Gear Games","Supergiant Games","Fatshark","Haemimont Games",
    "Owlcat Games","Tactical Adventures","Amplitude Studios",
]

PLATFORMS = ["Windows","Linux","Mac"]


def _random_tags(subgenre: str) -> list[str]:
    base = TAG_POOLS.get(subgenre, TAG_POOLS["crpg"])[:]
    extras = ["Fantasy","Dark","Magic","Multiplayer","Co-op","Singleplayer","Story Rich",
              "Atmospheric","Moody","Hand-drawn","Pixel Graphics","Retro","Cyberpunk"]
    random.shuffle(base)
    chosen = base[:random.randint(3, len(base))]
    chosen += random.sample(extras, random.randint(1, 4))
    return list(set(chosen))


def generate_synthetic(n: int = 500) -> pd.DataFrame:
    random.seed(42)
    np.random.seed(42)
    subgenres = list(TAG_POOLS.keys())
    rows = []

    used_names = list(RPG_NAMES)
    for i in range(n - len(RPG_NAMES)):
        used_names.append(f"RPG Adventure #{i+1}")

    random.shuffle(used_names)

    for i, name in enumerate(used_names[:n]):
        sub = random.choice(subgenres)
        tags = _random_tags(sub)
        dev  = random.choice(DEVELOPERS)
        price = random.choice([0, 4.99, 9.99, 14.99, 19.99, 29.99, 39.99, 49.99, 59.99])
        total_reviews = int(np.random.exponential(5000))
        positive_ratio = round(np.clip(np.random.normal(0.78, 0.12), 0.2, 0.99), 2)
        year = random.randint(2000, 2024)
        plats = random.sample(PLATFORMS, random.randint(1, 3))

        rows.append({
            "app_id":          100000 + i,
            "name":            name,
            "genres":          json.dumps(["RPG"]),
            "tags":            json.dumps(tags),
            "short_description": (
                f"An epic {sub.replace('_',' ')} set in a dark fantasy world. "
                f"Explore vast lands, fight powerful enemies and uncover ancient mysteries. "
                f"Developed by {dev}."
            ),
            "price":           price,
            "total_reviews":   total_reviews,
            "positive_ratio":  positive_ratio,
            "release_date":    f"{year}-01-01",
            "developer":       dev,
            "publisher":       dev,
            "platforms":       json.dumps(plats),
            "header_image":    "",
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[prepare_data] Dataset sintético guardado: {OUTPUT_CSV} ({len(df)} juegos)")
    return df


def download_kaggle() -> None:
    """
    Descarga el dataset de Steam de Kaggle.
    Requiere: pip install kaggle  +  ~/.kaggle/kaggle.json configurado.

    Dataset sugerido: 'fronkongames/steam-games-dataset'
    Después de descargar, filtra por tag RPG y guarda en src/data/steam_rpg_games.csv
    """
    try:
        import kaggle  # noqa
    except ImportError:
        print("Instala kaggle: pip install kaggle")
        return

    import subprocess
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", "fronkongames/steam-games-dataset",
        "-p", str(DATA_DIR), "--unzip"
    ], check=True)

    # Filtrar solo RPGs
    raw = DATA_DIR / "games.csv"
    if raw.exists():
        df = pd.read_csv(raw, low_memory=False)
        # columna 'Tags' en este dataset
        mask = df.get("Tags", pd.Series(dtype=str)).str.contains("RPG", case=False, na=False)
        rpg_df = df[mask].copy()
        rpg_df.to_csv(OUTPUT_CSV, index=False)
        print(f"[prepare_data] {len(rpg_df)} juegos RPG guardados en {OUTPUT_CSV}")
    else:
        print("[prepare_data] No se encontró games.csv tras la descarga.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["synthetic", "kaggle"], default="synthetic")
    parser.add_argument("--n", type=int, default=500, help="Número de juegos sintéticos")
    args = parser.parse_args()

    if args.mode == "synthetic":
        generate_synthetic(args.n)
    else:
        download_kaggle()
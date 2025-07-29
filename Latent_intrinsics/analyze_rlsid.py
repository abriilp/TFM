import os
import glob
from collections import defaultdict
from itertools import combinations

def analyze_rlsid_dataset(dataset_root, image_folder='Image'):
    """
    Analyze RLSID dataset to find scenes that share lighting conditions
    and can work together as validation scenes.

    Args:
        dataset_root: Path to dataset root directory
        image_folder: Folder name containing images ('Image', 'Reflectance', or 'Shading')

    Returns:
        Dictionary with analysis results, including auto-selected validation scenes
    """

    # Path to the image directory
    image_dir = os.path.join(dataset_root, image_folder)
    if not os.path.exists(image_dir):
        raise ValueError(f"Image directory not found: {image_dir}")

    # Get all image files
    image_files = glob.glob(os.path.join(image_dir, "*.png"))
    print(f"Found {len(image_files)} images in {image_dir}")

    # Parse all images and organize data
    scene_data = defaultdict(set)  # scene_background -> set of lighting conditions
    lighting_data = defaultdict(set)  # lighting_condition -> set of scene_backgrounds

    for img_path in image_files:
        try:
            filename = os.path.basename(img_path)
            # Parse filename: aaa_bbb_xxx_yyy_zzz.png
            name_parts = filename.replace('.png', '').split('_')
            if len(name_parts) >= 5:
                scene_idx = name_parts[0]  # aaa
                background = name_parts[1]  # bbb
                pan = name_parts[2]         # xxx
                tilt = name_parts[3]        # yyy
                color = name_parts[4]       # zzz

                scene_background = f"{scene_idx}_{background}"
                lighting_condition = f"{pan}_{tilt}_{color}"

                scene_data[scene_background].add(lighting_condition)
                lighting_data[lighting_condition].add(scene_background)

        except (IndexError, ValueError) as e:
            print(f"Skipping invalid filename: {filename} - {e}")
            continue

    print(f"\nFound {len(scene_data)} unique scenes")
    print(f"Found {len(lighting_data)} unique lighting conditions")

    # For each pair of scenes, find shared lighting conditions
    scene_pairs_with_shared = []
    scenes = list(scene_data.keys())

    for scene1, scene2 in combinations(scenes, 2):
        shared_lightings = scene_data[scene1].intersection(scene_data[scene2])
        if shared_lightings:
            scene_pairs_with_shared.append({
                'scene1': scene1,
                'scene2': scene2,
                'shared_lightings': shared_lightings,
                'count': len(shared_lightings)
            })

    # Sort by number of shared lighting conditions
    scene_pairs_with_shared.sort(key=lambda x: x['count'], reverse=True)

    # Group scenes by shared lighting
    def find_scene_groups(min_shared=1):
        scene_groups = []
        processed_scenes = set()

        for pair in scene_pairs_with_shared:
            if pair['count'] >= min_shared:
                scene1, scene2 = pair['scene1'], pair['scene2']

                found_group = None
                for group in scene_groups:
                    if scene1 in group['scenes'] or scene2 in group['scenes']:
                        found_group = group
                        break

                if found_group:
                    if scene1 not in found_group['scenes']:
                        if all(scene_data[scene1].intersection(scene_data[s]) for s in found_group['scenes']):
                            found_group['scenes'].add(scene1)
                    if scene2 not in found_group['scenes']:
                        if all(scene_data[scene2].intersection(scene_data[s]) for s in found_group['scenes']):
                            found_group['scenes'].add(scene2)
                else:
                    scene_groups.append({
                        'scenes': {scene1, scene2},
                        'shared_lightings': pair['shared_lightings']
                    })

        return scene_groups

    scene_groups = find_scene_groups(min_shared=1)

    validation_recommendations = []

    for group in scene_groups:
        scenes_list = sorted(list(group['scenes']))
        if len(scenes_list) >= 2:
            common_lightings = scene_data[scenes_list[0]]
            for scene in scenes_list[1:]:
                common_lightings = common_lightings.intersection(scene_data[scene])
            if len(common_lightings) > 0:
                validation_recommendations.append({
                    'scenes': scenes_list,
                    'common_lightings': len(common_lightings),
                    'scene_indices': [scene.split('_')[0] for scene in scenes_list]
                })

    for pair in scene_pairs_with_shared[:10]:  # top 10
        scene1_idx = pair['scene1'].split('_')[0]
        scene2_idx = pair['scene2'].split('_')[0]
        validation_recommendations.append({
            'scenes': [pair['scene1'], pair['scene2']],
            'common_lightings': pair['count'],
            'scene_indices': [scene1_idx, scene2_idx]
        })

    validation_recommendations.sort(key=lambda x: x['common_lightings'], reverse=True)

    # ---- NEW SECTION: Auto-select validation scenes (~20%) ----
    print("="*50)
    print("AUTO-SELECTION OF VALIDATION SCENES (~20%)")
    print("="*50)

    all_scenes = list(scene_data.keys())
    num_val_scenes = max(2, int(len(all_scenes) * 0.2))

    # Build scene graph where edges mean shared lighting
    scene_graph = defaultdict(set)
    for pair in scene_pairs_with_shared:
        s1 = pair['scene1']
        s2 = pair['scene2']
        if pair['count'] > 0:
            scene_graph[s1].add(s2)
            scene_graph[s2].add(s1)

    # Sort scenes by number of neighbors
    ranked_scenes = sorted(scene_graph.items(), key=lambda x: len(x[1]), reverse=True)

    selected = set()
    tried = set()
    for scene, neighbors in ranked_scenes:
        if scene in tried:
            continue
        tried.add(scene)
        available_neighbors = [n for n in neighbors if n not in selected]
        if available_neighbors:
            selected.add(scene)
            selected.add(available_neighbors[0])
        if len(selected) >= num_val_scenes:
            break

    val_scene_indices = sorted(set([s.split('_')[0] for s in selected]))
    print(f"Selected {len(selected)} scenes ({len(val_scene_indices)} unique indices)")
    print(f"Suggested validation_scenes = {val_scene_indices}")
    print(f"Scenes: {sorted(selected)}")
    print()

    return {
        'scene_data': dict(scene_data),
        'lighting_data': dict(lighting_data),
        'scene_pairs': scene_pairs_with_shared,
        'recommendations': validation_recommendations,
        'auto_validation_scenes': val_scene_indices
    }

# ---- Main runner ----

if __name__ == "__main__":
    dataset_root = "/data/storage/datasets/RLSID"  # Change this path to your dataset

    print("RLSID Dataset Validation Scene Analyzer")
    print("=" * 50)

    try:
        results = analyze_rlsid_dataset(dataset_root)

        print(f"\nAnalysis complete!")
        print(f"- Total scenes: {len(results['scene_data'])}")
        print(f"- Total lighting conditions: {len(results['lighting_data'])}")
        print(f"- Scene pairs with shared lighting: {len(results['scene_pairs'])}")
        print(f"- Auto-selected validation scenes: {results['auto_validation_scenes']}")

    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure to:")
        print("1. Set the correct dataset_root path")
        print("2. Ensure the Image folder exists in the dataset")
        print("3. Check that image files follow the naming convention: aaa_bbb_xxx_yyy_zzz.png")

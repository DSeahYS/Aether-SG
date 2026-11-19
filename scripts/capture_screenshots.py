import os
import time
from playwright.sync_api import sync_playwright

output_dir = r"C:\VSCode Folder\Ather-SG\docs\images"
os.makedirs(output_dir, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(
        executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        args=[
            "--headless=new",
            "--enable-webgl",
            "--use-gl=angle",
            "--use-angle=d3d11",
            "--no-sandbox",
            "--ignore-gpu-blocklist",
        ]
    )
    context = browser.new_context(
        viewport={"width": 1600, "height": 960},
        device_scale_factor=2
    )
    page = context.new_page()
    page.goto("http://127.0.0.1:8100/viewer/", wait_until="networkidle")
    
    # Wait for Three.js 3D models, GLTF mesh, and photos to settle
    time.sleep(5)
    
    # 1. Main tactical overview (Final winner route + 44 buildings + 118 photos)
    page.screenshot(path=os.path.join(output_dir, "01_tactical_overview.png"))
    print("Saved 01_tactical_overview.png")

    # 2. Evolution comparison - Gen 0 baseline
    gen0_btn = page.query_selector("#gen0Btn")
    if gen0_btn:
        gen0_btn.click()
        time.sleep(1.5)
        page.screenshot(path=os.path.join(output_dir, "02_evolution_gen0.png"))
        print("Saved 02_evolution_gen0.png")

    # Switch back to Final
    gen_final_btn = page.query_selector("#genFinalBtn")
    if gen_final_btn:
        gen_final_btn.click()
        time.sleep(1.5)

    # 3. Flood scenario (water level raised to 0.25 m)
    page.evaluate("() => { const s = document.querySelector('#floodSlider'); s.value = 0.25; s.dispatchEvent(new Event('input')); }")
    time.sleep(1.5)
    page.screenshot(path=os.path.join(output_dir, "03_flood_scenario.png"))
    print("Saved 03_flood_scenario.png")

    # Reset flood to 0.10 m
    page.evaluate("() => { const s = document.querySelector('#floodSlider'); s.value = 0.10; s.dispatchEvent(new Event('input')); }")
    time.sleep(1)

    # 4. Street View photo audit modal / inspection
    photos = page.query_selector_all("#filmScroll img")
    if photos:
        photos[0].click()
        time.sleep(1.5)
        page.screenshot(path=os.path.join(output_dir, "04_streetview_inspection.png"))
        print("Saved 04_streetview_inspection.png")

    browser.close()
    print("All screenshots captured successfully.")

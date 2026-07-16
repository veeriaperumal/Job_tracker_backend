import httpx

def test():
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    r = httpx.get("https://search.yahoo.com/search", params={"p": 'site:linkedin.com/jobs "Python Developer"'}, headers=headers)
    print("Status code:", r.status_code)
    html = r.text
    print("HTML contains class='algo':", 'class="algo' in html)
    print("HTML contains algo:", 'algo' in html)
    
    # Print occurrences of algo
    import re
    matches = [m.start() for m in re.finditer('algo', html)]
    print(f"Found {len(matches)} occurrences of 'algo'")
    for idx in matches[:5]:
        print(html[idx-50:idx+150])
        print("---")

if __name__ == '__main__':
    test()

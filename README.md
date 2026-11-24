# 🚇 MetroMate

An unofficial chatbot that helps you get to malls and universities in Doha by **Doha Metro** and the **Lusail Tram**.

**Live demo:** https://metromate-qtr.streamlit.app/


> **Disclaimer:** Unofficial student project, not affiliated with Qatar Rail or any university. Times are estimates, so please check with Qatar Rail before you travel.

## What it does

Tell MetroMate where you're starting (a metro or tram station) and where you want to go, it responds with:

- The fastest route, with stop counts, line changes, and an estimated time
- How to finish the trip from the closest station: walk (mentions walking distance if walkable) or taxi (mentions road distance)
- A free Metrolink bus option, with details only when you ask
- A warning when your route uses the Lusail Tram, since parts of it are suspended

It covers 47 places (malls, universities, and 15 buildings at Qatar University) and 59 metro and tram stations.

## How it works

Routes come from my own code (Dijkstra's algorithm on a graph of stations) and hand-collected data. The AI model (Groq's `gpt-oss-120b`) only understands the question, calls tools, and words the answer, so it never invents routes, distances, or buses. The rules, such as walk or taxi and which offer to make, are decided by tested code.

Other decisions: short first replies with details on request, honest "I don't have that" answers (no fares, timetables, or bus timing), small tool results to fit Groq's free limits, and usage caps for the public app.

## Run it yourself

You need Python 3.12+ and a free [Groq](https://console.groq.com) API key.

```bash
git clone https://github.com/FaiqaNaz001/MetroMate.git
cd MetroMate
pip install -r requirements.txt
```

Create a `.env` file (see `.env.example`) with `GROQ_API_KEY=your-key`, then:

```bash
python -m streamlit run app.py
```

## Tests

150+ automated tests that need no internet or API key:

```bash
python test_route_finder.py
python test_landmarks.py
python test_chat_tools.py
python test_chatbot.py
python test_web_helpers.py
```

## Limitations

- Starts from a station, not a place
- No tram timing, bus timing, fares, or timetables
- 47 places (only malls and universities), English only, and the AI's wording can occasionally vary

## Data and credits

- Stations and lines: Qatar Rail's published maps. Travel times: Moovit, Qatar Rail. Distances: Google Maps (November 2026). Bus routes: collected by hand, with source and date in `data/buses.csv`.
- Built with Python, Streamlit for the interface, and Groq AI, with AI assistance (Claude) for code and design guidance. The data was collected and verified by me.

**Faiqa Naz**, University of Doha for Science and Technology, Doha, Qatar · [GitHub](https://github.com/FaiqaNaz001) · [LinkedIn](https://www.linkedin.com/in/faiqa-naz-a56284346/)
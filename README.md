# Double Peak Generator

Interactive Streamlit application for generating and inspecting pairs of overlapping analytical peaks.

## Features

- Gaussian, Lorentzian, pseudo-Voigt and skew-normal peak families
- Independent-parameter and width-constrained generation modes
- Optional paired peak-family sampling
- Legacy CA/FA preset
- Independent peak-to-peak noise control
- Interactive preview of Peak 1, Peak 2 and their sum
- Wavelet preview with GD1, MH/GD2 and GD3
- Fractional wavelet scales, including values such as 0.5, 0.75, 1.5 and 3.7
- Dataset export as ZIP/NPZ/CSV/JSON

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push these files to a GitHub repository.
2. Open Streamlit Community Cloud.
3. Create a new app.
4. Select this GitHub repository and branch.
5. Set the main file path to `app.py`.
6. Deploy.

The application does not require model weights or external data files.

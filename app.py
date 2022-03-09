import streamlit as st
import streamlit.components.v1 as components
import base64
import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import io
import time
import joblib
import pandas as pd
from predict import predict_class

SAMPLING_RATE = 44_100 # default SR in Hz

st.set_page_config(
    page_title="Le Wagon Deep Dive",
    page_icon="\U0001F433", # whale emoji
    layout="wide",
    initial_sidebar_state="auto"
 )

# displaying sidebar
st.sidebar.image('images/LeWagonDeepDiveLogo.png', use_column_width='auto')
st.sidebar.markdown("""
                    # Le Wagon - Batch #802
                    [Victoria Metzger](https://www.linkedin.com/in/victoria-metzger-703416a7/)<br>
                    [Timothée Filhol](https://www.linkedin.com/in/timothee-filhol/)<br>
                    [Christian Lajouanie](https://www.linkedin.com/in/christianlajouanie/)
                    """,unsafe_allow_html=True)

st.title('Le Wagon - Deep Dive')

# upload the sound file
uploaded_sound = st.file_uploader('Load you file here',type=['wav'])

# javascript and CSS found here : https://github.com/mike-brady/Spectrogram-Player
my_css="""
    .sp-viewer {
    position: relative;
    background-repeat: no-repeat;
    }

    .sp-axis {
    position: absolute;
    }

    .sp-timeBar {
    width: 2px;
    height: 100%;
    position: absolute;
    left: 50%;
    background-color: red;
    }

    """

my_javascript ="""
    var spectrogram_player = {
    defaultWidth: 500,
    defaultHeight: 200,
    defaultFreqMin: 0,
    defaultFreqMax: 20,
    defaultAxisWidth: 30,
    defaultAxisDivisionHeight: 40,
    defaultAxisSmoothing: 2,

    playerIDs: [],

    init: function() {
        players = document.getElementsByClassName("spectrogram-player");
        for(i=0;i<players.length;i++) {
        player = players[i];

        imgElms = player.getElementsByTagName("img");
        if(imgElms.length == 0) {
            console.log('Spectrogram Player: Missing image element');
            continue;
        } else if(imgElms.length > 1) {
            console.log('Spectrogram Player: Found multiple images in player. First image element is assumed to be the spectrogram.')
        }

        audioElms = player.getElementsByTagName("audio");
        if(audioElms.length == 0) {
            console.log('Spectrogram Player: Missing audio element');
            continue;
        } else if(audioElms.length != 1) {
            console.log('Spectrogram Player: Found multiple audio elements in player. First audio element is assumed to be the audio file.')
        }

        width = (player.getAttribute('data-width')) ? player.getAttribute('data-width') : this.defaultWidth;
        height = (player.getAttribute('data-height')) ? player.getAttribute('data-height') : this.defaultHeight;
        freqMin = (player.getAttribute('data-freq-min')) ? player.getAttribute('data-freq-min') : this.defaultFreqMin;
        freqMax = (player.getAttribute('data-freq-max')) ? player.getAttribute('data-freq-max') : this.defaultFreqMax;
        axisWidth = (player.getAttribute('data-axis-width')) ? player.getAttribute('data-axis-width') : this.defaultAxisWidth;
        axisDivisionHeight = (player.getAttribute('data-axis-division-height')) ? player.getAttribute('data-axis-division-height') : this.defaultAxisDivisionHeight;
        axisSmoothing = (player.getAttribute('data-axis-smoothing')) ? player.getAttribute('data-axis-smoothing') : this.defaultAxisSmoothing;

        spectrogram = imgElms[0].src;
        imgElms[0].parentNode.removeChild(imgElms[0]);

        audio = audioElms[0];
        audio.id = "sp-audio"+i;
        audio.style.width = width+"px";

        //Create viewer element
        viewer = document.createElement('div');
        viewer.className = "sp-viewer";
        viewer.id = "sp-viewer"+i;

        viewer.style.width = width+"px";
        viewer.style.height = height+"px";

        viewer.style.backgroundImage = "url('"+spectrogram+"')";
        viewer.style.backgroundPosition = width/2+"px";
        viewer.style.backgroundSize = "auto "+height+"px";

        if(axisWidth > 0) {
            divisions = Math.floor(height/axisDivisionHeight);
            if(axisSmoothing != 0)
            divisions = this.smoothAxis(freqMax-freqMin, divisions, [0,.5,.25], axisSmoothing);

            axis = this.drawAxis(axisWidth,height,freqMin,freqMax,divisions,"kHz");
            axis.className = "sp-axis";
            viewer.appendChild(axis);
        }

        timeBar = document.createElement('div');
        timeBar.className = "sp-timeBar";
        viewer.appendChild(timeBar);

        player.insertBefore(viewer, player.firstChild);

        this.playerIDs.push(i);
        }

        setInterval(function() { spectrogram_player.moveSpectrograms(); },33);
    },

    moveSpectrograms: function() {
        for(i=0;i<this.playerIDs.length;i++) {
        id = this.playerIDs[i];
        audio = document.getElementById("sp-audio"+id);
        if(audio.paused)
            continue;

        viewer = document.getElementById("sp-viewer"+id);
        viewerWidth = viewer.offsetWidth;
        duration = audio.duration;

        viewerStyle = viewer.currentStyle || window.getComputedStyle(viewer, false);
        img = new Image();
        //remove url(" and ") from backgroundImage string
        img.src = viewerStyle.backgroundImage.replace(/url\(\"|\"\)$/ig, '');
        //get the width of the spectrogram image based on its scaled size * its native size
        spectWidth = viewer.offsetHeight/img.height*img.width;

        viewer.style.backgroundPosition = viewerWidth/2 - audio.currentTime/duration*spectWidth + "px";
        }
    },

    smoothAxis: function(range, baseDivision, allowedDecimals, distance) {
        if(distance==0)
        return baseDivision;

        subtractFirst = (distance<0) ? false : true;

        for(var i=0;i<=distance;i++) {
        d1 = (subtractFirst) ? baseDivision-i : baseDivision+i;
        d2 = (subtractFirst) ? baseDivision+i : baseDivision-i;

        if(d1 > 0) {
            decimal = this.qoutientDecimal(range, d1, 4)
            if(allowedDecimals.indexOf(decimal) > -1)
            return d1;
        }

        if(d2 > 0) {
            decimal = this.qoutientDecimal(range, d2, 4)
            if(allowedDecimals.indexOf(decimal) > -1)
            return d2;
        }
        }

        return baseDivision;
    },

    drawAxis: function(width,height,min,max,divisions,unit) {
        axis = document.createElement('canvas');
        axis.width = width;
        axis.height = height;

        ctx = axis.getContext("2d");

        ctx.fillStyle ="rgba(0,0,0,.1)";
        ctx.fillRect(0,0,width,height);

        ctx.font = "12px Arial";
        ctx.textAlign = "right";
        ctx.textBaseline = "top";
        ctx.fillStyle ="rgb(100,100,100)";
        ctx.strokeStyle ="rgb(100,100,100)";

        range = max-min;

        for(var i=0;i<divisions;i++) {
        y = Math.round(height/divisions*i);
        ctx.moveTo(0,y+.5);
        ctx.lineTo(width,y+.5);
        ctx.stroke();

        curVal = (divisions-i) * range/divisions + min*1;

        ctx.fillText(Math.round(curVal*100)/100,width,y);
        }

        ctx.textBaseline = "bottom";
        ctx.fillText(unit,width,height);

        return axis;
    },

    qoutientDecimal: function(dividend, divisor, precision) {
        quotient = dividend/divisor;

        if(precision === undefined)
        b = 1;
        else
        b = Math.pow(10,precision);

        return Math.round(quotient%1 *b)/b;
    }
    };

    window.onload = function() { spectrogram_player.init(); };

    """


# when an audio file is loaded
if uploaded_sound is not None:

    # converting uploaded_sound to be correctly understood in the html component
    sound_url = base64.b64encode(uploaded_sound.getvalue()).decode("utf-8")

    # converting sound in spectrogram
    audio_data, sr = librosa.load(uploaded_sound,sr=SAMPLING_RATE)
    mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    plt.figure(figsize=(30,5))
    img = librosa.display.specshow(mel_spec_db, cmap = 'plasma')

    # save fig in a buffer
    buffer = io.BytesIO()
    plt.savefig(buffer,bbox_inches='tight',pad_inches=0)
    plt.close()

    # converting buffer to be correctly understood in the html component
    image_url = base64.b64encode(buffer.getvalue()).decode("utf-8")


    with st.expander('Spectrogram Player', expanded=True):
        #html to display the spectrogram player
        components.html(
            f"""
            <style>{my_css}</style>
            <script type="text/javascript">{my_javascript}</script>

            <div class="spectrogram-player" data-width=1300 data-height=300 data-freq-min=0 data-freq-max=8>
                <img src="data:image/png;base64,{image_url}">
                <audio controls controlsList="nodownload">
                    <source src="data:audio/wav;base64,{sound_url}" type="audio/wav">
                </audio>
            </div>
            &copy; <a href="https://github.com/mike-brady/Spectrogram-Player">Mike Brady</a></span>
            """,
            height=400)


    # loading model while user is playing with spectrogram player
    model = joblib.load('model.joblib')
    df_species=pd.read_csv('species_table.csv')

    # after 'Predict' button is clicked
    if st.button('Predict'):

        with st.spinner('Prediction in progress...'):
            col1, col2, col3 = st.columns(3)

            with col1:
                placeholder1=st.info('...')
                placeholder2=st.image('images/species.gif')

            with col2:
                placeholder3=st.info('...')
                placeholder4=st.image('images/species.gif')

            with col3:
                placeholder5=st.info('...')
                placeholder6=st.image('images/species.gif')

            prediction=predict_class(audio_data, model)
            class_proba = [item for item in prediction.items()]
            time.sleep(5)

            with col1:
                placeholder1.info(f'{df_species.iloc[class_proba[2][0]].common_name} with a confidence of {class_proba[2][1]} %.')
                placeholder2.image('images_species/'+df_species.iloc[class_proba[2][0]].image_name)

            with col2:
                placeholder3.info(f'{df_species.iloc[class_proba[1][0]].common_name} with a confidence of {class_proba[1][1]} %.')
                placeholder4.image('images_species/'+df_species.iloc[class_proba[1][0]].image_name)

            with col3:
                placeholder5.info(f'{df_species.iloc[class_proba[0][0]].common_name} with a confidence of {class_proba[0][1]} %.')
                placeholder6.image('images_species/'+df_species.iloc[class_proba[0][0]].image_name)

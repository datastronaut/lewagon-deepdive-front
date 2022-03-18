###################################
##          DEEP DIVE            ##
##   LE WAGON CODING BOOTCAMP    ##
##          BATCH #802           ##
##      January - March 2022     ##
##                               ##
##  Developers                   ##
##   - Victoria Metzger          ##
##   - Timothée Filhol           ##
##   - Christian Lajouanie       ##
##                               ##
###################################

# imports
import io
import time
import base64
import joblib
import librosa
import numpy as np
import pandas as pd
import librosa.display
import streamlit as st
import matplotlib.pyplot as plt
from predict import predict_class
import streamlit.components.v1 as components

# default SR in Hz
SAMPLING_RATE = 44_100

# web page configuration
st.set_page_config(
    page_title="Le Wagon Deep Dive",
    page_icon="\U0001F433", # whale emoji
    layout="wide",
    initial_sidebar_state="auto")

# sidebar
st.sidebar.markdown("""
        # Le Wagon - Batch #802
        [Victoria Metzger](https://www.linkedin.com/in/victoria-metzger-703416a7/)<br>
        [Timothée Filhol](https://www.linkedin.com/in/timothee-filhol/)<br>
        [Christian Lajouanie](https://www.linkedin.com/in/christianlajouanie/)
        """, unsafe_allow_html=True)
st.sidebar.image('images/qrcode.png', width=100)
st.sidebar.markdown("""
        This app is our final project at [Le Wagon](https://www.lewagon.com/).<br>
        We have trained our model thanks to the amazing work of William Watkins \
        and the thousands of recordings available on the<br>\
        [Watkins Marine Mammal Sound Database](https://cis.whoi.edu/science/B/whalesounds/index.cfm)
        """,unsafe_allow_html=True)
st.sidebar.video('https://youtu.be/vwwDGtuFbSQ?t=4045')
st.sidebar.caption('Our Demo Day presentation (starts at 1:07:25)')
st.sidebar.markdown("""
        Github repositories:
        [Front-end](https://github.com/ChristianDesCodes/lewagon-deepdive-front) /
        [Back-end](https://github.com/ChristianDesCodes/lewagon-deepdive)
                    """)


# main title
st.image('images/LeWagonDeepDiveLogo.png', width=100)
"""
# Deep Dive
## Predict marine mammal species from their song
"""

# uploader
uploaded_sound = st.file_uploader('',type=['wav'])

# when audio file is loaded
if uploaded_sound is not None:

    # convert uploaded_sound to be correctly understood in the html code below
    sound_url = base64.b64encode(uploaded_sound.getvalue()).decode("utf-8")

    # convert sound in spectrogram
    audio_data, sr = librosa.load(uploaded_sound,sr=SAMPLING_RATE)
    mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    plt.figure(figsize=(30,5))
    img = librosa.display.specshow(mel_spec_db, cmap = 'plasma')

    # save fig in a buffer
    buffer = io.BytesIO()
    plt.savefig(buffer,bbox_inches='tight',pad_inches=0)
    plt.close()

    # convert buffered image to be correctly understood in the html code below
    image_url = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # display spectrogram player in an expander
    with st.expander('Spectrogram', expanded=True):

        # code for spectrogram player found here:
        # https://github.com/mike-brady/Spectrogram-Player

        # css
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

        # javascript
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

        # html
        my_html = f"""
            <style>{my_css}</style>
            <script type="text/javascript">{my_javascript}</script>

            <div class="spectrogram-player" data-width=600 data-height=300 data-freq-min=0 data-freq-max=8>
                <img src="data:image/png;base64,{image_url}">
                <audio controls controlsList="nodownload">
                    <source src="data:audio/wav;base64,{sound_url}" type="audio/wav">
                </audio>
            </div>
        """

        # display player on the app
        components.html(my_html,height=370)
        st.caption('Spectrogram player developped by [Mike Brady](https://github.com/mike-brady/Spectrogram-Player)')



    # load model
    model = joblib.load('model.joblib')
    df_species=pd.read_csv('species/species_table.csv')

    # button for prediction
    if st.button('Find the species'):

        # make prediction with gif animation
        with st.spinner('Prediction in progress...'):
            col1, col2, col3 = st.columns(3)

            with col1:
                placeholder11=st.empty()
                placeholder12=st.empty()
                placeholder13=st.image('images/species.gif')

            with col2:
                placeholder21=st.empty()
                placeholder22=st.empty()
                placeholder23=st.image('images/species.gif')

            with col3:
                placeholder31=st.empty()
                placeholder32=st.empty()
                placeholder33=st.image('images/species.gif')

            prediction=predict_class(audio_data, model)
            class_proba = [item for item in prediction.items()]
            time.sleep(1)

        # display results
        with col1:
            placeholder11.image('images/gold.png', width=100)
            placeholder12.info(f'**{class_proba[2][1]}%**  -  {df_species.iloc[class_proba[2][0]].common_name}')
            placeholder13.image('species/'+df_species.iloc[class_proba[2][0]].image_name)

        with col2:
            placeholder21.image('images/silver.png', width=100)
            placeholder22.info(f'**{class_proba[1][1]}%**  -  {df_species.iloc[class_proba[1][0]].common_name}')
            placeholder23.image('species/'+df_species.iloc[class_proba[1][0]].image_name)

        with col3:
            placeholder31.image('images/bronze.png', width=100)
            placeholder32.info(f'**{class_proba[0][1]}%**  -  {df_species.iloc[class_proba[0][0]].common_name}')
            placeholder33.image('species/'+df_species.iloc[class_proba[0][0]].image_name)

        # easter egg
        if 'mystery' in uploaded_sound.name:
            with st.expander('Reveal', expanded=False):
                st.audio('sound/star-wars-cantina-song.mp3',start_time=0)
                st.image('images/chewbacca.png',width=300)

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import base64
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

'''
# Le Wagon - Deep Dive project
'''

# # ouvrir une image à partir d'une url:
# url = 'https://media.fisheries.noaa.gov/styles/original/s3/dam-migration/640x427-minke-whale.png'
# response = requests.get(url, stream=True)
# response.raw.decode_content = True
# image = Image.open(response.raw)

# displaying a cover picture
image = Image.open('images/cover.png')
st.image(image, use_column_width=False, width = 500 )

# upload the sound file
uploaded_sound = st.file_uploader('Load you file here',type=['wav','mp3'])

# javascript and CSS found here : https://github.com/mike-brady/Spectrogram-Player
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

my_css="""
.sp-viewer {
  position: relative;
  background-repeat: no-repeat;
}

.sp-axis {
  position: absolute;
}

.sp-timeBar {
  width: 1px;
  height: 100%;
  position: absolute;
  left: 50%;
  background-color: #f00020;
}

"""


if uploaded_sound is not None:

    # converting uploaded_sound to be correctly understood in html component
    sound_url = base64.b64encode(uploaded_sound.getvalue()).decode("utf-8")

    # converting sound in spectrogram
    y, sr = librosa.load(uploaded_sound,sr=44_100)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    plt.figure(figsize=(50,5))
    img = librosa.display.specshow(mel_spec_db, cmap = 'plasma')

    plt.savefig('images/spectrogram.png',bbox_inches='tight',pad_inches=0)

    # converting img to be correctly understood in html component
    file_ = open('images/spectrogram.png', "rb")
    contents = file_.read()
    image_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    data_width=600
    data_height=200
    data_freq_min=0
    data_freq_max=8


    components.html(
    f"""
    <style>{my_css}</style>
    <script type="text/javascript">{my_javascript}</script>

    <div class="spectrogram-player" data-width=600 data-height=200 data-freq-min=0 data-freq-max=8>
        <img src="data:image/png;base64,{image_url}">
        <audio controls controlsList="nodownload">
            <source src="data:audio/wav;base64,{sound_url}" type="audio/wav">
        </audio>
    </div>
    Spectrogram<br />
    &copy; <a href="https://github.com/mike-brady/Spectrogram-Player">Mike Brady</a></span>
    """,
    height=600,
    scrolling=True
)

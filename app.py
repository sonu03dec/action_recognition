from flask import Flask, render_template, redirect, url_for, request, send_from_directory, session, flash
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm, Form
from wtforms import StringField, PasswordField, BooleanField, FileField, SubmitField, ValidationError
from wtforms.validators import InputRequired, Email, Length
from flask_sqlalchemy  import SQLAlchemy
from werkzeug import SharedDataMiddleware
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from moviepy.editor import VideoFileClip, concatenate
import numpy as np
import os
from keras.models import load_model
import cv2

app = Flask(__name__)
app.config['SECRET_KEY'] = 'Thisissupposedtobesecret!'
SAVE_FOLDER = 'E:/projects/SV_Match_Highlights/saved/'
app.config['SAVE_FOLDER'] = SAVE_FOLDER

bootstrap = Bootstrap(app)

class UploadForm(Form):
    style={'class': 'ourClasses', 'style': 'background-image: linear-gradient(#022127, #37304c);border:none;font-family: "Roboto", sans-serif;font-size: 16px;font-weight: 400;color:white;padding-left:20px;padding-right:20px;'}
    video_file = FileField('')
    submit = SubmitField('Upload',render_kw=style)

    def validate_video_file(self, field):
        if field.data.filename[-4:].lower() != '.mp4':
            raise ValidationError('Invalid file extension')

@app.route('/', methods=['GET', 'POST'])
def index():
    video = None
    form = UploadForm()
    if form.validate_on_submit():
        video = 't_upload/tennis/' + form.video_file.data.filename
        form.video_file.data.save(os.path.join(app.static_folder, video))
        filename = form.video_file.data.filename
        model = load_model("E:/projects/SV_Tennis/LRCNNTennis.h5")


        classes = ['backhand', 'forehand']
        cap = cv2.VideoCapture("E:/projects/SV_Tennis/static/t_upload/tennis/" + filename)

        FILE_OUTPUT = "E:/projects/SV_Tennis/saved/" + filename + "cut.mp4"
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter(FILE_OUTPUT, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                              10, (frame_width, frame_height))


        count = 0
        #Q = deque(maxlen= "size")
        while cap.isOpened():
            _, frame = cap.read()
            
            pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            
            if frame is not None:
                output = frame.copy()
                frame_count = ("Frames:{}".format(count))
                count= count + 1
            else:
                print("sorry frames was ****completed****")
                break
            
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(frame, ((360//2)*6, 640//2), interpolation = cv2.INTER_CUBIC)
            frame1 = resized.reshape(1, 6, 320, 180, 3)
            
            #resized = cv2.resize(frame, (256, 256), interpolation = cv2.INTER_CUBIC)
            #frame1 = resized.reshape(1, 256, 256, 3)

            
            pred_array = model.predict(frame1)
            print(pred_array)

            result = classes[np.argmax(pred_array)]
            
            
            score = float("%0.2f" % (max(pred_array[0]) * 100))

            
            text1 = ("Activity:{} |".format(result))
            cv2.putText(output, text1, (35, 50), cv2.FONT_HERSHEY_PLAIN, 1.25, (255, 255, 255), 2)
            cv2.rectangle(output, (20, 30), (550, 60), color=(0, 255, 0), thickness=2)
            
            text = ("Score:{} |".format(score))
            cv2.putText(output, text, (250, 50), cv2.FONT_HERSHEY_PLAIN, 1.25, (255, 255, 255), 2)
            cv2.putText(output, frame_count, (400, 50), cv2.FONT_HERSHEY_PLAIN, 1.25, (0, 255, 155), 2)
            
            print(f'Result: {result}, Score: {score}')
            
            cv2.imshow("frame", output)
            
            out.write(output)
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break


        cap.release()
        out.release()
        cv2.destroyAllWindows()
        return redirect(url_for('index'))

    return render_template('index.html', form=form, video=video)




if __name__ == '__main__':
    app.run(debug=True)
FROM python:3.8

WORKDIR /app

RUN mkdir -p templates

COPY app.py trained_salary_pred_LinearReg_model.joblib requirements.txt Dockerfile ./
ADD templates templates

RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 3000

CMD python ./app.py
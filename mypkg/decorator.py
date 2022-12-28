import smtplib
from email.mime.text import MIMEText
from email.utils import formatdate
import traceback
import sys


def error_gmail(f):
    def wrapper(*args,**kwargs):
        #SMTPのオブジェクト作成。GmailのSMTPポートは587
        smtpobj = smtplib.SMTP('smtp.gmail.com', 587)

        #メールサーバに対する応答
        smtpobj.ehlo()
        #暗号化通信開始
        smtpobj.starttls()
        smtpobj.ehlo()
        #ログイン
        smtpobj.login("konni10chiwa@gmail.com", "qzafodromntuhvri")

        #送信元、送信先
        mail_from = "konni10chiwa@gmail.com"
        mail_to = "konni10chiwa@gmail.com"



        try:
            val = f(*args,**kwargs)
            # #本文
            # text = "pythonスクリプトが正常に完了しました"

            # #メッセージのオブジェクト
            # msg = MIMEText(text)
            # msg['Subject'] = "pythonスクリプト完了"
            # msg['From'] = mail_from
            # msg['To'] = mail_to
            # msg['Date'] = formatdate(localtime=True)
            # smtpobj.sendmail(mail_from, mail_to, msg.as_string())

            return val
        except Exception as error:
            if type(error) == KeyboardInterrupt:
                print("キーボード中断")
                sys.exit()
                

            error_log = traceback.format_exc()
            print("エラー発生")
            print(error_log)
            #本文
            text = f"pythonスクリプトでエラーが発生しました\ntype:{type(error)}\n{str(error)}\n\n詳細\n{error_log}"

            #メッセージのオブジェクト
            msg = MIMEText(text)
            msg['Subject'] = "pythonスクリプトでエラーが発生しました。"
            msg['From'] = mail_from
            msg['To'] = mail_to
            msg['Date'] = formatdate(localtime=True)
            smtpobj.sendmail(mail_from, mail_to, msg.as_string())

            return 0
        
    return wrapper

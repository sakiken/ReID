from PyQt5.Qt import *
import sys

app = QApplication(sys.argv)

w = QWidget()
w.resize(210, 280)
palette = QPalette()
pix=QPixmap("./assert/135.jpg")

pix = pix.scaled(w.width(),w.height())

palette.setBrush(QPalette.Background,QBrush(pix))
w.setPalette(palette)

w.show()

if __name__ == '__main__':
    sys.exit(app.exec_())

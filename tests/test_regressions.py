import io
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest

import foto_para_desenho as engine
from image_input import MAX_INPUT_PIXELS, MAX_UPLOAD_BYTES, prepare_image

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "assets" / "demo_fotoimagem.jpg"


def encoded(image, image_format="PNG", **kwargs):
    buffer = io.BytesIO()
    image.save(buffer, format=image_format, **kwargs)
    return buffer.getvalue()


class ImageInputTests(unittest.TestCase):
    def test_invalid_file_and_limits(self):
        for data in (b"invalid image", b"x" * (MAX_UPLOAD_BYTES + 1)):
            with self.subTest(size=len(data)), self.assertRaises(ValueError):
                prepare_image(data)
        oversized = Image.new("RGB", (4001, 4000))
        self.assertGreater(oversized.width * oversized.height, MAX_INPUT_PIXELS)
        with self.assertRaisesRegex(ValueError, "16 megapixels"):
            prepare_image(encoded(oversized))
        with self.assertRaisesRegex(ValueError, "JPG ou PNG"):
            prepare_image(encoded(Image.new("RGB", (32, 32)), "GIF"))

    def test_phone_orientation_and_resize(self):
        image = Image.new("RGB", (2000, 1000), "red")
        exif = Image.Exif()
        exif[274] = 6
        result, resized = prepare_image(encoded(image, "JPEG", exif=exif))
        self.assertTrue(resized)
        self.assertEqual(result.size, (800, 1600))
        self.assertEqual(result.mode, "RGB")

    def test_transparent_image_uses_white_background(self):
        result, resized = prepare_image(encoded(Image.new("RGBA", (32, 32), (0, 0, 0, 0))))
        self.assertEqual(result.getpixel((0, 0)), (255, 255, 255))
        self.assertFalse(resized)


class EffectControlTests(unittest.TestCase):
    def test_hq_cleanup_removes_small_contours_and_keeps_large_shapes(self):
        image = np.full((256, 256, 3), 220, dtype=np.uint8)
        cv2.rectangle(image, (20, 20), (160, 220), (30, 30, 30), 3)
        cv2.circle(image, (210, 100), 3, (30, 30, 30), -1)
        detailed = engine.fx_hq(image, min_area=0)
        clean = engine.fx_hq(image, min_area=80)
        self.assertFalse(np.array_equal(detailed, clean))
        self.assertTrue(np.array_equal(detailed[10:230, 10:170], clean[10:230, 10:170]))
        self.assertLess(np.count_nonzero(clean[90:110, 200:220] == 0), np.count_nonzero(detailed[90:110, 200:220] == 0))

    def test_skin_strength_changes_binary_line_output(self):
        image = cv2.imread(str(DEMO))
        image = cv2.resize(image, (640, 640))
        weak = engine.fx_colorir_linhas_pro_v2(image, skin_strength=0.0)
        stronger = engine.fx_colorir_linhas_pro_v2(image, skin_strength=0.55)
        self.assertFalse(np.array_equal(weak, stronger))
        disabled = engine.fx_colorir_linhas_pro_v2(image, skin_suppress=False, skin_strength=0.55)
        self.assertTrue(np.array_equal(weak, disabled))
        self.assertEqual(set(np.unique(stronger)), {0, 255})


class AppFlowTests(unittest.TestCase):
    def app(self):
        return AppTest.from_file(str(ROOT / "app.py"), default_timeout=45).run()

    def assert_healthy(self, app):
        self.assertEqual([e.message for e in app.exception], [])
        self.assertEqual([e.value for e in app.error], [])

    def test_demo_all_fifteen_effects_and_png_download(self):
        app = self.app()
        self.assert_healthy(app)
        self.assertEqual(len(app.selectbox[0].options), 15)
        for effect in app.selectbox[0].options:
            with self.subTest(effect=effect):
                app.selectbox[0].select(effect).run()
                app.button[0].click().run()
                self.assert_healthy(app)
                self.assertEqual(len(app.image), 2)
                self.assertEqual(len(app.download_button), 1)
                with Image.open(io.BytesIO(app.session_state["result"]["png"])) as output:
                    self.assertEqual(output.format, "PNG")
                    self.assertLessEqual(max(output.size), 1600)

    def test_result_survives_rerun_and_is_cleared_for_changed_selection(self):
        app = self.app()
        app.button[0].click().run()
        png = app.session_state["result"]["png"]
        app.run()
        self.assertEqual(app.session_state["result"]["png"], png)
        self.assertEqual(len(app.download_button), 1)
        app.slider[0].set_value(0.8).run()
        self.assertEqual(len(app.download_button), 0)
        app.radio[0].set_value("Enviar minha imagem").run()
        self.assertTrue(app.button[0].disabled)

    def test_valid_and_corrupt_upload(self):
        app = self.app()
        app.radio[0].set_value("Enviar minha imagem").run()
        app.file_uploader[0].upload("exemplo.jpg", DEMO.read_bytes(), "image/jpeg").run()
        app.button[0].click().run()
        self.assert_healthy(app)
        self.assertEqual(len(app.download_button), 1)
        app.file_uploader[0].upload("corrompido.png", b"not an image", "image/png").run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(app.button[0].disabled)
        self.assertIn("JPG ou PNG válido", app.error[0].value)
        self.assertEqual(len(app.download_button), 0)


if __name__ == "__main__":
    unittest.main()

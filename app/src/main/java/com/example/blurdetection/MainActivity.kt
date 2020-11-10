package com.example.blurdetection

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.drawable.BitmapDrawable
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import com.example.blurdetection.ml.BlurModelQuant
import kotlinx.android.synthetic.main.activity_main.*
import org.tensorflow.lite.support.image.TensorImage
import java.io.IOException

class MainActivity : AppCompatActivity() {

    private val LOAD_IMAGE = 1

    private lateinit var bitmap: Bitmap

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Permission to read external storage
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestPermissions(arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE), 1)
        }

        load_image.setOnClickListener {
             result_text.text = ""

            val i = Intent(
                Intent.ACTION_PICK,
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI
            )
            startActivityForResult(i, LOAD_IMAGE)
        }

        detect_image.setOnClickListener {
            status_image.visibility = View.VISIBLE
            status_image.setImageResource(R.drawable.loading_animation)

            try {

                // Read the image as Bitmap
                bitmap = (imageview.getDrawable() as BitmapDrawable).bitmap

                // We reshape the image into 400*400
                bitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)

                // Load the model file
                val model = BlurModelQuant.newInstance(baseContext)

                // Creates inputs for reference.
                val image = TensorImage.fromBitmap(bitmap)

                // Runs model inference and gets result.
                val outputs = model.process(image)
                    .probabilityAsCategoryList.apply {
                        sortByDescending { it.score }
                    }.take(2)

                // Create an empty string
                var outputString = ""

                //Take the highest result
                val result = outputs[0]
                outputString = result.label

                runOnUiThread {
                    result_text.text = translate(outputString)
                    status_image.visibility = View.GONE
                }

                // Releases model resources if no longer used.
                model.close()

            } catch (e: IOException) {
                status_image.visibility = View.VISIBLE
                status_image.setImageResource(R.drawable.ic_broken_img)
                finish()
            }
        }

    }

    fun translate(value: String): String? {
        if (value == "sharp") return "Good"
        if (value == "defocused_blurred") return "Warning: The Image is not Clear"
        return if (value == "motion_blurred") "Warning: The Image is not Clear" else ""
    }

    @SuppressLint("Recycle")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        // This functions return the selected image from gallery
        if (requestCode == LOAD_IMAGE && resultCode == Activity.RESULT_OK && null != data) {
            val selectedImage = data.data
            val filePathColumn = arrayOf(MediaStore.Images.Media.DATA)
            if (BuildConfig.DEBUG && selectedImage == null) {
                error("Assertion failed")
            }
            val cursor = contentResolver.query(
                selectedImage!!,
                filePathColumn, null, null, null
            )!!
            cursor.moveToFirst()
            val columnIndex = cursor.getColumnIndex(filePathColumn[0])
            val picturePath = cursor.getString(columnIndex)
            cursor.close()
            imageview.setImageBitmap(BitmapFactory.decodeFile(picturePath))

            //Setting the URI so we can read the Bitmap from the image
            imageview.setImageURI(null)
            imageview.setImageURI(selectedImage)
        }
    }
}
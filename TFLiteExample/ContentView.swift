import SwiftUI
import TensorFlowLite
import Foundation

struct ContentView: View {
    
    @State var output = "Hello"
    @State var celsius = ""
    @State var fahrenheit: Float = 0.0
    
    var body: some View {
        VStack {
            TextField( "Enter celsius value", text: $celsius)
            Button("Infar"){
                predictFahrenheit()
            }
            Text("\(fahrenheit) Fahrenheit ")
        }
        .padding()
    }
    
    func predictFahrenheit() {
        // Load the TFLite model
        guard let modelPath = Bundle.main.path(forResource: "model", ofType: "tflite") else {
            fatalError("Model not found")
        }
        
        do {
            // Initialize an interpreter with the model.
            let interpreter = try Interpreter(modelPath: modelPath)
            
            // Allocate memory for the model's input `Tensor`s.
            try interpreter.allocateTensors()
            
            let inputData: Data   // Should be initialized
            
            // Process input data
            inputData = withUnsafeBytes(of: Float(celsius) ?? 0.0) { Data($0) }
            try interpreter.copy(inputData, toInputAt: 0)
            
            // Run inference by invoking the `Interpreter`.
            try interpreter.invoke()
            
            // Get the output `Tensor`
            let outputTensor = try interpreter.output(at: 0)
            let outputData = outputTensor.data
            
            // Process inference data
            fahrenheit = outputData.withUnsafeBytes { $0.load(as: Float.self) }
            
        } catch let error {
            print("Error: \(error)")
            
        }
    }
}


#Preview {
    ContentView()
}

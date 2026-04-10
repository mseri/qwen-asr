#!/usr/bin/swift

import AVFoundation
import CoreAudio
import Foundation

// --- Configuration ---
let outputSampleRate: Double = 16000.0  // 44100.0
let outputChannels: UInt32 = 1
let outputBitsPerSample: UInt16 = 16

// --- Helper: Logging to Stderr ---
// We must use stderr for text so stdout is kept clean for binary audio data
func log(_ message: String) {
  fputs("\(message)\n", stderr)
}

func readInput() -> String? {
  return readLine(strippingNewline: true)
}

// --- CoreAudio Device Selection ---

struct AudioDevice {
  let id: AudioDeviceID
  let name: String
  let inputChannels: Int
}

func getAvailableInputDevices() -> [AudioDevice] {
  var propertyAddress = AudioObjectPropertyAddress(
    mSelector: kAudioHardwarePropertyDevices,
    mScope: kAudioObjectPropertyScopeGlobal,
    mElement: kAudioObjectPropertyElementMain
  )

  var dataSize: UInt32 = 0
  let status = AudioObjectGetPropertyDataSize(
    AudioObjectID(kAudioObjectSystemObject), &propertyAddress, 0, nil, &dataSize)

  guard status == noErr else {
    log("Error getting device list size.")
    return []
  }

  let deviceCount = Int(dataSize) / MemoryLayout<AudioDeviceID>.size
  var deviceIDs = [AudioDeviceID](repeating: 0, count: deviceCount)

  let status2 = AudioObjectGetPropertyData(
    AudioObjectID(kAudioObjectSystemObject), &propertyAddress, 0, nil, &dataSize, &deviceIDs)

  guard status2 == noErr else {
    log("Error getting device list data.")
    return []
  }

  var inputDevices: [AudioDevice] = []

  for id in deviceIDs {
    // 1. Get Name
    var nameSize = UInt32(MemoryLayout<CFString>.size)
    var deviceName: CFString = "" as CFString
    var nameAddr = AudioObjectPropertyAddress(
      mSelector: kAudioObjectPropertyName,
      mScope: kAudioObjectPropertyScopeGlobal,
      mElement: kAudioObjectPropertyElementMain
    )
    AudioObjectGetPropertyData(id, &nameAddr, 0, nil, &nameSize, &deviceName)

    // 2. Check for Input Channels
    var streamAddr = AudioObjectPropertyAddress(
      mSelector: kAudioDevicePropertyStreamConfiguration,
      mScope: kAudioDevicePropertyScopeInput,
      mElement: 0
    )

    var streamSize: UInt32 = 0
    AudioObjectGetPropertyDataSize(id, &streamAddr, 0, nil, &streamSize)

    let bufferListSize = UnsafeMutablePointer<Int>.allocate(capacity: 1)
    // We create a buffer list just to check capacity
    let bufferList = UnsafeMutablePointer<AudioBufferList>.allocate(capacity: Int(streamSize))

    // Actually reading the stream config to count channels
    var actualSize = streamSize
    let statusStream = AudioObjectGetPropertyData(id, &streamAddr, 0, nil, &actualSize, bufferList)

    var totalChannels = 0
    if statusStream == noErr {
      let buffers = UnsafeMutableAudioBufferListPointer(bufferList)
      for buffer in buffers {
        totalChannels += Int(buffer.mNumberChannels)
      }
    }

    bufferList.deallocate()
    bufferListSize.deallocate()

    if totalChannels > 0 {
      inputDevices.append(
        AudioDevice(id: id, name: deviceName as String, inputChannels: totalChannels))
    }
  }

  return inputDevices
}

// --- The Audio Engine ---

class AudioStreamer {
  let engine = AVAudioEngine()
  var converter: AVAudioConverter?
  let targetFormat: AVAudioFormat
  let stdout = FileHandle.standardOutput

  init() {
    guard
      let format = AVAudioFormat(
        commonFormat: .pcmFormatInt16,
        sampleRate: outputSampleRate,
        channels: AVAudioChannelCount(outputChannels),
        interleaved: true)
    else {
      log("Critical Error: Could not define output format.")
      exit(1)
    }
    self.targetFormat = format
  }

  func start() {
    // 1. Get Devices and Prompt User
    let devices = getAvailableInputDevices()
    if devices.isEmpty {
      log("No input devices found!")
      exit(1)
    }

    log("--- Available Input Devices ---")
    for (index, device) in devices.enumerated() {
      log("[\(index)] \(device.name) (\(device.inputChannels) channels)")
    }

    log("Select a device number:")
    guard let inputStr = readInput(), let index = Int(inputStr), index >= 0 && index < devices.count
    else {
      log("Invalid selection.")
      exit(1)
    }

    let selectedDevice = devices[index]
    log("Selected: \(selectedDevice.name)")

    // 2. Configure Engine with Selected Device
    let inputNode = engine.inputNode  // Accessing this creates the singleton node

    // We must reach into the underlying AudioUnit to set the hardware device ID
    let inputUnit = inputNode.audioUnit!
    var deviceID = selectedDevice.id
    let error = AudioUnitSetProperty(
      inputUnit,
      kAudioOutputUnitProperty_CurrentDevice,
      kAudioUnitScope_Global,
      0,
      &deviceID,
      UInt32(MemoryLayout<AudioDeviceID>.size))

    if error != noErr {
      log("Error setting AudioUnit device: \(error)")
      exit(1)
    }

    // 3. Set up format conversion
    let inputFormat = inputNode.inputFormat(forBus: 0)
    log("Device Format: \(inputFormat.sampleRate)Hz, \(inputFormat.channelCount) channels")

    if inputFormat.sampleRate == 0 {
      log("Error: Device reports 0Hz sample rate. Check permissions.")
      exit(1)
    }

    guard let converter = AVAudioConverter(from: inputFormat, to: targetFormat) else {
      log("Error: Could not create audio converter.")
      exit(1)
    }
    self.converter = converter

    // 4. Write Header and Start
    writeWavHeader()

    inputNode.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) {
      [weak self] (buffer, time) in
      guard let self = self else { return }
      self.process(buffer: buffer)
    }

    do {
      try engine.start()
      log("Streaming audio to stdout... (Press Ctrl+C to stop)")
      CFRunLoopRun()  // Block forever
    } catch {
      log("Error starting engine: \(error)")
      exit(1)
    }
  }

  func process(buffer inputBuffer: AVAudioPCMBuffer) {
    let inputCallback: AVAudioConverterInputBlock = { inNumPackets, outStatus in
      outStatus.pointee = .haveData
      return inputBuffer
    }

    let ratio = targetFormat.sampleRate / inputBuffer.format.sampleRate
    let capacity = UInt32(Double(inputBuffer.frameLength) * ratio)

    guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: capacity)
    else { return }

    var error: NSError? = nil
    let status = converter?.convert(to: outputBuffer, error: &error, withInputFrom: inputCallback)

    if status != .error, let channelData = outputBuffer.int16ChannelData {
      let bytesLength = Int(outputBuffer.frameLength) * Int(targetFormat.channelCount) * 2
      let data = Data(bytes: channelData[0], count: bytesLength)
      try? stdout.write(contentsOf: data)
    }
  }

  func writeWavHeader() {
    var header = Data()
    header.append("RIFF".data(using: .ascii)!)
    header.append(UInt32(UInt32.max).data)
    header.append("WAVE".data(using: .ascii)!)
    header.append("fmt ".data(using: .ascii)!)
    header.append(UInt32(16).data)
    header.append(UInt16(1).data)
    header.append(UInt16(outputChannels).data)
    header.append(UInt32(outputSampleRate).data)
    let byteRate = UInt32(outputSampleRate) * outputChannels * UInt32(outputBitsPerSample / 8)
    header.append(byteRate.data)
    let blockAlign = UInt16(outputChannels * UInt32(outputBitsPerSample / 8))
    header.append(blockAlign.data)
    header.append(outputBitsPerSample.data)
    header.append("data".data(using: .ascii)!)
    header.append(UInt32(UInt32.max).data)
    try? stdout.write(contentsOf: header)
  }
}

extension Numeric {
  var data: Data {
    var source = self
    return Data(bytes: &source, count: MemoryLayout<Self>.size)
  }
}

let streamer = AudioStreamer()
streamer.start()

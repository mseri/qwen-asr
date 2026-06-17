#!/usr/bin/swift

import AVFoundation
import CoreAudio
import Foundation

let outputSampleRate: Double = 16000.0
let outputChannels: UInt32 = 1
let outputBitsPerSample: UInt16 = 16

func log(_ message: String) {
  fputs("\(message)\n", stderr)
}

func readInput() -> String? {
  readLine(strippingNewline: true)
}

enum StreamError: Error, CustomStringConvertible {
  case message(String)

  var description: String {
    switch self {
    case .message(let message):
      return message
    }
  }
}

enum CaptureSource: String {
  case input = "in"
  case output = "out"
  case both = "both"
}

struct Options {
  let source: CaptureSource
}

func usage() {
  log("Usage: mic2wav.swift [--source in|out|both]")
}

func parseOptions() -> Options {
  var source: CaptureSource = .input
  var index = 1
  let args = CommandLine.arguments

  while index < args.count {
    let arg = args[index]
    switch arg {
    case "--source":
      index += 1
      guard index < args.count, let parsed = CaptureSource(rawValue: args[index]) else {
        usage()
        exit(1)
      }
      source = parsed
    case "-h", "--help":
      usage()
      exit(0)
    default:
      log("Unknown option: \(arg)")
      usage()
      exit(1)
    }
    index += 1
  }

  return Options(source: source)
}

func check(_ status: OSStatus, _ operation: String) throws {
  guard status == noErr else {
    throw StreamError.message("\(operation) failed: \(status)")
  }
}

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
    var nameSize = UInt32(MemoryLayout<CFString?>.size)
    var nameAddr = AudioObjectPropertyAddress(
      mSelector: kAudioObjectPropertyName,
      mScope: kAudioObjectPropertyScopeGlobal,
      mElement: kAudioObjectPropertyElementMain
    )
    let nameStorage = UnsafeMutableRawPointer.allocate(
      byteCount: Int(nameSize), alignment: MemoryLayout<CFString?>.alignment)
    defer { nameStorage.deallocate() }
    AudioObjectGetPropertyData(id, &nameAddr, 0, nil, &nameSize, nameStorage)
    let deviceName = nameStorage.assumingMemoryBound(to: CFString?.self).pointee ?? "" as CFString

    var streamAddr = AudioObjectPropertyAddress(
      mSelector: kAudioDevicePropertyStreamConfiguration,
      mScope: kAudioDevicePropertyScopeInput,
      mElement: kAudioObjectPropertyElementMain
    )

    var streamSize: UInt32 = 0
    AudioObjectGetPropertyDataSize(id, &streamAddr, 0, nil, &streamSize)

    guard streamSize > 0 else { continue }

    let bufferList = UnsafeMutablePointer<AudioBufferList>.allocate(capacity: Int(streamSize))
    defer { bufferList.deallocate() }

    var actualSize = streamSize
    let statusStream = AudioObjectGetPropertyData(id, &streamAddr, 0, nil, &actualSize, bufferList)

    var totalChannels = 0
    if statusStream == noErr {
      let buffers = UnsafeMutableAudioBufferListPointer(bufferList)
      for buffer in buffers {
        totalChannels += Int(buffer.mNumberChannels)
      }
    }

    if totalChannels > 0 {
      inputDevices.append(
        AudioDevice(id: id, name: deviceName as String, inputChannels: totalChannels))
    }
  }

  return inputDevices
}

func selectInputDevice() throws -> AudioDevice {
  let devices = getAvailableInputDevices()
  guard !devices.isEmpty else {
    throw StreamError.message("No input devices found.")
  }

  log("--- Available Input Devices ---")
  for (index, device) in devices.enumerated() {
    log("[\(index)] \(device.name) (\(device.inputChannels) channels)")
  }

  log("Select a device number:")
  guard let inputStr = readInput(), let index = Int(inputStr), index >= 0, index < devices.count
  else {
    throw StreamError.message("Invalid selection.")
  }

  let selected = devices[index]
  log("Selected: \(selected.name)")
  return selected
}

func getTapFormat(_ tapID: AudioObjectID) throws -> AVAudioFormat {
  var format = AudioStreamBasicDescription()
  var formatSize = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
  var formatAddress = AudioObjectPropertyAddress(
    mSelector: kAudioTapPropertyFormat,
    mScope: kAudioObjectPropertyScopeGlobal,
    mElement: kAudioObjectPropertyElementMain
  )

  try check(
    AudioObjectGetPropertyData(tapID, &formatAddress, 0, nil, &formatSize, &format),
    "Querying tap format")

  guard let audioFormat = AVAudioFormat(streamDescription: &format) else {
    throw StreamError.message("Could not build AVAudioFormat from tap format.")
  }

  return audioFormat
}

final class PCMWriter {
  private let stdout = FileHandle.standardOutput
  private let lock = NSLock()

  init() {
    writeWavHeader()
  }

  func write(samples: UnsafePointer<Int16>, count: Int) {
    guard count > 0 else { return }
    let data = Data(bytes: samples, count: count * MemoryLayout<Int16>.size)
    lock.lock()
    defer { lock.unlock() }
    try? stdout.write(contentsOf: data)
  }

  private func writeWavHeader() {
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
    lock.lock()
    defer { lock.unlock() }
    try? stdout.write(contentsOf: header)
  }
}

final class Int16RingBuffer {
  private var buffer: [Int16]
  private var readIndex = 0
  private var writeIndex = 0
  private var storedCount = 0
  private let lock = NSLock()

  init(capacity: Int) {
    buffer = [Int16](repeating: 0, count: max(1, capacity))
  }

  func write(samples: UnsafePointer<Int16>, count: Int) {
    guard count > 0 else { return }

    lock.lock()
    defer { lock.unlock() }

    let capacity = buffer.count
    var start = 0
    var sampleCount = count

    if sampleCount >= capacity {
      start = sampleCount - capacity
      sampleCount = capacity
      readIndex = 0
      writeIndex = 0
      storedCount = 0
    }

    let overflow = max(0, storedCount + sampleCount - capacity)
    if overflow > 0 {
      readIndex = (readIndex + overflow) % capacity
      storedCount -= overflow
    }

    for idx in 0..<sampleCount {
      buffer[writeIndex] = samples[start + idx]
      writeIndex = (writeIndex + 1) % capacity
    }
    storedCount += sampleCount
  }

  func read(into destination: UnsafeMutablePointer<Int16>, count: Int) -> Int {
    guard count > 0 else { return 0 }

    lock.lock()
    defer { lock.unlock() }

    let toRead = min(count, storedCount)
    guard toRead > 0 else { return 0 }

    let capacity = buffer.count
    for idx in 0..<toRead {
      destination[idx] = buffer[readIndex]
      readIndex = (readIndex + 1) % capacity
    }
    storedCount -= toRead
    return toRead
  }
}

final class MixedSink {
  private let outputRing: Int16RingBuffer
  private let writer: PCMWriter
  private var scratch: [Int16] = []

  init(outputRing: Int16RingBuffer, writer: PCMWriter) {
    self.outputRing = outputRing
    self.writer = writer
  }

  func writeMixed(micSamples: UnsafePointer<Int16>, count: Int) {
    guard count > 0 else { return }
    if scratch.count < count {
      scratch = [Int16](repeating: 0, count: count)
    }

    scratch.withUnsafeMutableBufferPointer { outputBuffer in
      guard let base = outputBuffer.baseAddress else { return }
      let outputCount = outputRing.read(into: base, count: count)

      for idx in 0..<count {
        let mic = Int32(micSamples[idx])
        let out = idx < outputCount ? Int32(base[idx]) : 0
        let mixed = max(Int32(Int16.min), min(Int32(Int16.max), mic + out))
        base[idx] = Int16(mixed)
      }

      writer.write(samples: base, count: count)
    }
  }
}

final class AudioSampleSink {
  let handler: (UnsafePointer<Int16>, Int) -> Void

  init(_ handler: @escaping (UnsafePointer<Int16>, Int) -> Void) {
    self.handler = handler
  }

  func consume(_ samples: UnsafePointer<Int16>, count: Int) {
    handler(samples, count)
  }
}

final class MicrophoneCapture {
  private let engine = AVAudioEngine()
  private let targetFormat: AVAudioFormat
  private let sink: AudioSampleSink
  private var converter: AVAudioConverter?

  init(targetFormat: AVAudioFormat, sink: AudioSampleSink) {
    self.targetFormat = targetFormat
    self.sink = sink
  }

  func start() throws {
    let selectedDevice = try selectInputDevice()
    let inputNode = engine.inputNode

    guard let inputUnit = inputNode.audioUnit else {
      throw StreamError.message("Could not access input AudioUnit.")
    }

    var deviceID = selectedDevice.id
    try check(
      AudioUnitSetProperty(
        inputUnit,
        kAudioOutputUnitProperty_CurrentDevice,
        kAudioUnitScope_Global,
        0,
        &deviceID,
        UInt32(MemoryLayout<AudioDeviceID>.size)),
      "Setting input device")

    let inputFormat = inputNode.inputFormat(forBus: 0)
    log("Microphone Format: \(inputFormat.sampleRate)Hz, \(inputFormat.channelCount) channels")

    guard inputFormat.sampleRate > 0 else {
      throw StreamError.message("Input device reports 0Hz sample rate. Check permissions.")
    }

    guard let converter = AVAudioConverter(from: inputFormat, to: targetFormat) else {
      throw StreamError.message("Could not create microphone audio converter.")
    }
    self.converter = converter

    inputNode.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { [weak self] buffer, _ in
      self?.process(buffer: buffer)
    }

    do {
      try engine.start()
    } catch {
      throw StreamError.message("Error starting microphone engine: \(error)")
    }
  }

  func stop() {
    engine.inputNode.removeTap(onBus: 0)
    engine.stop()
  }

  private func process(buffer inputBuffer: AVAudioPCMBuffer) {
    guard let converter = converter else { return }

    let inputCallback: AVAudioConverterInputBlock = { _, outStatus in
      outStatus.pointee = .haveData
      return inputBuffer
    }

    let ratio = targetFormat.sampleRate / inputBuffer.format.sampleRate
    let capacity = max(1, Int(ceil(Double(inputBuffer.frameLength) * ratio)) + 64)

    guard
      let outputBuffer = AVAudioPCMBuffer(
        pcmFormat: targetFormat, frameCapacity: AVAudioFrameCount(capacity))
    else {
      return
    }

    var error: NSError?
    let status = converter.convert(to: outputBuffer, error: &error, withInputFrom: inputCallback)

    guard status != .error, let channelData = outputBuffer.int16ChannelData else { return }
    let frameCount = Int(outputBuffer.frameLength)
    guard frameCount > 0 else { return }
    sink.consume(channelData[0], count: frameCount)
  }
}

@available(macOS 14.2, *)
final class SystemOutputCapture {
  private let targetFormat: AVAudioFormat
  private let sink: AudioSampleSink
  private let callbackQueue = DispatchQueue(label: "qwen-asr.system-output")

  private var tapID: AudioObjectID = kAudioObjectUnknown
  private var aggregateID: AudioDeviceID = kAudioObjectUnknown
  private var ioProcID: AudioDeviceIOProcID?
  private var sourceFormat: AVAudioFormat?
  private var converter: AVAudioConverter?

  init(targetFormat: AVAudioFormat, sink: AudioSampleSink) {
    self.targetFormat = targetFormat
    self.sink = sink
  }

  func start() throws {
    let tapDescription = CATapDescription(monoGlobalTapButExcludeProcesses: [])
    tapDescription.name = "qwen-asr System Output"
    tapDescription.uuid = UUID()
    tapDescription.isPrivate = true
    tapDescription.isExclusive = true
    tapDescription.muteBehavior = .unmuted

    var createdTapID = AudioObjectID(kAudioObjectUnknown)
    try check(AudioHardwareCreateProcessTap(tapDescription, &createdTapID), "Creating process tap")
    tapID = createdTapID

    let tapUID = tapDescription.uuid.uuidString
    let aggregateDescription: [String: Any] = [
      kAudioAggregateDeviceNameKey as String: "qwen-asr System Output",
      kAudioAggregateDeviceUIDKey as String: "antirez.qwen-asr.mic2wav.\(UUID().uuidString)",
      kAudioAggregateDeviceIsPrivateKey as String: true,
      kAudioAggregateDeviceTapAutoStartKey as String: false,
      kAudioAggregateDeviceTapListKey as String: [
        [
          kAudioSubTapUIDKey as String: tapUID,
          kAudioSubTapDriftCompensationKey as String: true,
        ]
      ],
    ]

    var createdAggregateID = AudioDeviceID(kAudioObjectUnknown)
    do {
      try check(
        AudioHardwareCreateAggregateDevice(aggregateDescription as CFDictionary, &createdAggregateID),
        "Creating aggregate device")
    } catch {
      AudioHardwareDestroyProcessTap(createdTapID)
      tapID = kAudioObjectUnknown
      throw error
    }
    aggregateID = createdAggregateID

    let sourceFormat = try getTapFormat(tapID)
    self.sourceFormat = sourceFormat

    guard let converter = AVAudioConverter(from: sourceFormat, to: targetFormat) else {
      throw StreamError.message("Could not create system output audio converter.")
    }
    self.converter = converter

    try installIOProc()
    try check(AudioDeviceStart(aggregateID, ioProcID), "Starting aggregate device")
  }

  func stop() {
    if aggregateID != kAudioObjectUnknown, let ioProcID {
      AudioDeviceStop(aggregateID, ioProcID)
      AudioDeviceDestroyIOProcID(aggregateID, ioProcID)
    }
    ioProcID = nil

    if aggregateID != kAudioObjectUnknown {
      AudioHardwareDestroyAggregateDevice(aggregateID)
      aggregateID = kAudioObjectUnknown
    }

    if tapID != kAudioObjectUnknown {
      AudioHardwareDestroyProcessTap(tapID)
      tapID = kAudioObjectUnknown
    }
  }

  private func installIOProc() throws {
    var createdIOProcID: AudioDeviceIOProcID?
    try check(
      AudioDeviceCreateIOProcIDWithBlock(&createdIOProcID, aggregateID, callbackQueue) {
        [weak self] _, inputData, _, _, _ in
        self?.process(inputData: inputData)
      },
      "Creating IOProc")
    ioProcID = createdIOProcID
  }

  private func process(inputData: UnsafePointer<AudioBufferList>?) {
    guard
      let inputData,
      let sourceFormat,
      let converter
    else {
      return
    }

    let firstBuffer = inputData.pointee.mBuffers
    guard firstBuffer.mData != nil, firstBuffer.mDataByteSize > 0 else { return }

    let bytesPerFrame = Int(sourceFormat.streamDescription.pointee.mBytesPerFrame)
    guard bytesPerFrame > 0 else { return }

    let frameCount = Int(firstBuffer.mDataByteSize) / bytesPerFrame
    guard frameCount > 0 else { return }

    let mutableBufferList = UnsafeMutablePointer<AudioBufferList>(mutating: inputData)
    guard
      let inputBuffer = AVAudioPCMBuffer(
        pcmFormat: sourceFormat, bufferListNoCopy: mutableBufferList, deallocator: nil)
    else {
      return
    }
    inputBuffer.frameLength = AVAudioFrameCount(frameCount)

    let ratio = targetFormat.sampleRate / sourceFormat.sampleRate
    let capacity = max(1, Int(ceil(Double(frameCount) * ratio)) + 64)
    guard
      let outputBuffer = AVAudioPCMBuffer(
        pcmFormat: targetFormat, frameCapacity: AVAudioFrameCount(capacity))
    else {
      return
    }

    let inputCallback: AVAudioConverterInputBlock = { _, outStatus in
      outStatus.pointee = .haveData
      return inputBuffer
    }

    var error: NSError?
    let status = converter.convert(to: outputBuffer, error: &error, withInputFrom: inputCallback)

    guard status != .error, let channelData = outputBuffer.int16ChannelData else { return }
    let outputFrameCount = Int(outputBuffer.frameLength)
    guard outputFrameCount > 0 else { return }

    sink.consume(channelData[0], count: outputFrameCount)
  }
}

final class AppController {
  private let options: Options
  private let writer = PCMWriter()
  private let targetFormat: AVAudioFormat
  private var microphoneCapture: MicrophoneCapture?
  private var systemOutputCapture: AnyObject?
  private var signalSource: DispatchSourceSignal?

  init(options: Options) {
    self.options = options
    guard
      let format = AVAudioFormat(
        commonFormat: .pcmFormatInt16,
        sampleRate: outputSampleRate,
        channels: AVAudioChannelCount(outputChannels),
        interleaved: true)
    else {
      fatalError("Could not create target output format.")
    }
    self.targetFormat = format
  }

  func start() throws {
    installSignalHandler()

    switch options.source {
    case .input:
      let sink = AudioSampleSink { [writer] samples, count in
        writer.write(samples: samples, count: count)
      }
      let mic = MicrophoneCapture(targetFormat: targetFormat, sink: sink)
      try mic.start()
      microphoneCapture = mic
      log("Streaming microphone audio to stdout... (Press Ctrl+C to stop)")

    case .output:
      guard #available(macOS 14.2, *) else {
        throw StreamError.message("--source out requires macOS 14.2 or newer.")
      }

      let sink = AudioSampleSink { [writer] samples, count in
        writer.write(samples: samples, count: count)
      }
      let outputCapture = SystemOutputCapture(targetFormat: targetFormat, sink: sink)
      try outputCapture.start()
      systemOutputCapture = outputCapture
      log("Streaming system audio output to stdout... (Press Ctrl+C to stop)")

    case .both:
      guard #available(macOS 14.2, *) else {
        throw StreamError.message("--source both requires macOS 14.2 or newer.")
      }

      let outputRing = Int16RingBuffer(capacity: Int(outputSampleRate * 10.0))
      let mixer = MixedSink(outputRing: outputRing, writer: writer)

      let outputSink = AudioSampleSink { samples, count in
        outputRing.write(samples: samples, count: count)
      }
      let outputCapture = SystemOutputCapture(targetFormat: targetFormat, sink: outputSink)
      try outputCapture.start()
      systemOutputCapture = outputCapture

      let micSink = AudioSampleSink { samples, count in
        mixer.writeMixed(micSamples: samples, count: count)
      }
      let mic = MicrophoneCapture(targetFormat: targetFormat, sink: micSink)
      do {
        try mic.start()
      } catch {
        outputCapture.stop()
        systemOutputCapture = nil
        throw error
      }
      microphoneCapture = mic
      log("Streaming mixed microphone + system audio to stdout... (Press Ctrl+C to stop)")
    }

    CFRunLoopRun()
  }

  func stop() {
    microphoneCapture?.stop()
    microphoneCapture = nil

    if #available(macOS 14.2, *), let outputCapture = systemOutputCapture as? SystemOutputCapture {
      outputCapture.stop()
    }
    systemOutputCapture = nil

    signalSource?.cancel()
    signalSource = nil
  }

  private func installSignalHandler() {
    signal(SIGINT, SIG_IGN)
    let source = DispatchSource.makeSignalSource(signal: SIGINT, queue: .main)
    source.setEventHandler { [weak self] in
      self?.stop()
      exit(0)
    }
    source.resume()
    signalSource = source
  }
}

extension Numeric {
  var data: Data {
    var source = self
    return Data(bytes: &source, count: MemoryLayout<Self>.size)
  }
}

let options = parseOptions()

do {
  let app = AppController(options: options)
  try app.start()
} catch {
  log("\(error)")
  exit(1)
}

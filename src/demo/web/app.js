// Few-Shot KWS web UI

const API = '';

// State
let audioBlobs = { enroll: null, detect: null, long: null };
let detectionHistory = [];
let mediaRecorder = null;
let audioCtx = null;
let analyser = null;
let recAnimFrame = null;
let modelProfilesState = { active: null, profiles: [], canRebuildOnSwitch: false, loading: true };
let pendingModelProfileId = null;
let currentLang = localStorage.getItem('kws_lang') || 'en';
let lastModelInfo = null;
let enrolledKeywordsState = new Set();
let openSetCalibrationCache = {};

const OPEN_SET_SPLITS = {
  gsc_17_17: {
    label: 'GSC Open-Set 17/17',
    known: ['yes', 'stop', 'happy', 'bird', 'dog', 'tree', 'marvin', 'four', 'learn', 'wow', 'sheila', 'zero', 'down', 'left', 'right', 'off', 'three'],
    unknown: ['no', 'go', 'up', 'on', 'one', 'two', 'five', 'six', 'seven', 'eight', 'nine', 'bed', 'cat', 'house', 'backward', 'forward', 'follow'],
    heldout: ['visual'],
  },
};

const I18N = {
  en: {
    skipMain: 'Skip to main content',
    switchCheckpoint: 'Switch checkpoint',
    chooseEnrollmentHandling: 'Choose enrollment handling',
    closeModelSwitchDialog: 'Close model switch dialog',
    switchRebuild: 'Switch and rebuild enrollment',
    switchClear: 'Switch and clear enrollment',
    cancel: 'Cancel',
    primaryNavigation: 'Primary navigation',
    sectionNavigation: 'Section navigation',
    mobileAppHeader: 'Mobile app header',
    brandSub: 'Research operations console',
    mobileBrandSub: 'Keyword lab',
    main: 'Main',
    advanced: 'Advanced',
    enrollment: 'Enrollment',
    detection: 'Detection',
    detect: 'Detect',
    longFile: 'Long File',
    streaming: 'Streaming',
    openSetTest: 'Open-Set Test',
    openSet: 'Open-Set',
    modelInfo: 'Model Info',
    info: 'Info',
    modelLoaded: 'Model loaded',
    sidebarCaption: 'DSCNN-L - Prototypical Net',
    ready: 'Ready',
    activeCheckpoint: 'Active checkpoint',
    loadingModel: 'Loading model...',
    reloadProfiles: 'Reload profiles',
    loadingCheckpoints: 'Loading checkpoints...',
    modelCheckpointSwitcher: 'Model checkpoint switcher',
    kwsConsole: 'KWS Research Console',
    commandCenter: 'Keyword Command Center',
    heroText: 'Enroll a compact support set, calibrate open-set thresholds, and test static or streaming keyword detection from one workspace.',
    fewShotEnrollment: 'Few-shot enrollment',
    openSetRejection: 'Open-set rejection',
    streamingTest: 'Streaming test',
    prototypeCalibration: 'Prototype calibration',
    bestCheckpoint: 'Best checkpoint',
    gscDataset: 'GSC Dataset',
    quickPreset: 'Quick preset',
    keywords: 'Keywords',
    enrollFromGsc: 'Enroll from GSC',
    microphone: 'Microphone',
    keywordName: 'Keyword name',
    clickToRecord: 'Click to record ~1s',
    uploadFile: 'Upload file',
    profiles: 'Profiles',
    profileName: 'Profile name',
    save: 'Save',
    load: 'Load',
    enrolledKeywords: 'Enrolled Keywords',
    clearAll: 'Clear all',
    activityLog: 'Activity Log',
    actionsHere: 'Actions will appear here',
    detectionDesc: 'Upload or record ~1 second of audio to identify enrolled keywords',
    audioInput: 'Audio Input',
    chooseDetectionAudio: 'Choose detection audio file',
    dropAudio: 'Drop audio file here or click to browse',
    dropAudioSub: 'WAV, MP3, OGG - about 1 second',
    recordDetectionSample: 'Record detection sample',
    orRecordWithMic: 'Or record with mic',
    settings: 'Settings',
    detectionThreshold: 'Detection threshold',
    usePerClassThreshold: 'Use per-class threshold',
    closeWordGuard: 'Close-word guard',
    closeWordGuardHelp: 'Reject uncertain matches when top-1 and top-2 words are too close.',
    closeWordGuardOn: 'Guard ON',
    closeWordGuardOff: 'Guard OFF',
    acceptMargin: 'Accept margin',
    perClassOn: 'Per-class ON',
    perClassOff: 'Per-class OFF',
    detectKeyword: 'Detect Keyword',
    uploadThenDetect: 'Upload or record audio, then click Detect',
    distanceToPrototypes: 'Distance to prototypes',
    mfccFeatures: 'MFCC Features',
    detectionHistory: 'Detection History',
    clear: 'Clear',
    noDetectionsYet: 'No detections yet',
    longFileDetection: 'Long File Detection',
    longFileDesc: 'Upload a longer audio file (5-30s) for multi-word detection',
    audioFile: 'Audio File',
    chooseLongAudio: 'Choose long audio file',
    dropLongAudio: 'Drop long audio file (5-30 seconds)',
    threshold: 'Threshold',
    segmentationMethod: 'Segmentation method',
    energyBased: 'Energy-based',
    sileroVad: 'Silero VAD',
    minDuration: 'Min duration (ms)',
    perClassThreshold: 'Per-class threshold',
    distanceMetric: 'Distance metric',
    detectWords: 'Detect Words',
    groundTruth: 'Ground Truth',
    optional: 'Optional',
    gtHelp: 'Upload a TXT file to check accuracy after detection',
    chooseGroundTruth: 'Choose ground truth file',
    dropTxt: 'Drop .txt file here',
    gtFormat: 'Format: expected_keyword (one per line, matching segment order)',
    txtFormat: 'TXT format:',
    gtFormatHelp: 'Comma-separated or one per line. Order matches detected segments.',
    labelsInFile: 'Labels in file',
    labelCount: '{count} labels',
    labelsLoaded: 'Loaded {count} labels from {name}',
    moreLabels: '+{count} more',
    invalidGroundTruth: 'No labels found in the ground-truth file',
    timingJson: 'Timing JSON',
    timingHelp: 'Optional: upload .timings.json to compare detections by time overlap',
    chooseTimingJson: 'Choose timing JSON file',
    dropTimingJson: 'Drop .timings.json here',
    timingsInFile: 'Timings in file',
    timingCount: '{count} timings',
    timingsLoaded: 'Loaded {count} timings from {name}',
    invalidTimingJson: 'No valid timings found in the JSON file',
    timingMatchMode: 'Timing match mode: Expected is selected by maximum time overlap with each detection.',
    expectedTimeline: 'Expected timing',
    detectedTimeline: 'Detected timing',
    segments: 'Segments',
    correct: 'Correct',
    accuracy: 'Accuracy',
    uploadLongToAnalyze: 'Upload a long audio file to analyze',
    streamingTitle: 'Real-Time Streaming',
    streamingDesc: 'Live microphone keyword detection with sliding window',
    startStreaming: 'Start Streaming',
    stopStreaming: 'Stop Streaming',
    listening: 'Listening...',
    stopped: 'Stopped',
    streamingStarted: 'Streaming started',
    streamingStopped: 'Streaming stopped - {count} detections',
    webSocketError: 'WebSocket error',
    liveStats: 'Live Stats',
    detections: 'Detections',
    lastKeyword: 'Last Keyword',
    elapsed: 'Elapsed',
    detectionFeed: 'Detection Feed',
    startStreamingEmpty: 'Start streaming to see live detections',
    openSetTitle: 'Open-Set Rejection Test',
    openSetDesc: 'Verify that non-enrolled words are correctly rejected as "unknown"',
    openSetPreset: 'Open-set preset',
    manualSplit: 'Manual split',
    knownWords: 'Known/enrolled GSC words',
    unknownWords: 'Unknown GSC words to test',
    openSetSplitSummary: 'Known: {known} words. Unknown: {unknown} words. Holdout: {heldout}.',
    samplesPerWord: 'Samples per word',
    acceptMargin: 'Accept margin',
    runTest: 'Run Test',
    runOpenSetTest: 'Run Open-Set Test',
    calibrateOpenSet: 'Calibrate Open-Set',
    randomSeed: 'Random seed',
    openSetInputHelp: 'Known words are the candidate labels. Unknown words are sampled from local GSC folders and must be rejected as "unknown".',
    runningOpenSet: 'Running open-set test...',
    runningCalibration: 'Running calibration...',
    knownTested: 'Known tested',
    unknownTested: 'Unknown tested',
    candidateLabels: 'Candidate labels',
    balancedScore: 'Balanced score',
    openSetAcc: 'Open-set ACC',
    keywordAcc: 'Keyword ACC',
    unknownRejectAcc: 'Unknown reject ACC',
    falseAcceptRate: 'False accept rate',
    falseRejectRate: 'False reject rate',
    falseAccepts: 'False accepts',
    knownMisses: 'Known misses',
    noFalseAccepts: 'No false accepts',
    noKnownMisses: 'No known misses',
    sourceWord: 'Source word',
    file: 'File',
    skippedUnknownWords: 'Skipped unknown words',
    missingKnownWords: 'Missing known words',
    missingUnknownWords: 'Missing unknown words',
    shortAudioWords: 'Not enough samples',
    openSetEvaluated: 'Open-set test done: {accuracy} open-set ACC',
    openSetCalibrated: 'Calibration done: {score} balanced score',
    calibrationResults: 'Calibration results',
    bestBalanced: 'Best balanced',
    bestRejectUnknown: 'Best reject unknown',
    bestRecognizeKeyword: 'Best recognize keyword',
    applySettings: 'Apply settings',
    settingsApplied: 'Calibration settings applied',
    thresholdValue: 'Threshold',
    perClass: 'Per-class',
    policy: 'Policy',
    language: 'Language',
    comingSoon: 'Coming soon - requires GSC data',
    resultsHere: 'Results will appear here',
    modelInformation: 'Model Information',
    modelInformationDesc: 'Architecture details and evaluation results',
    modelCheckpoint: 'Model Checkpoint',
    fallbackSelector: 'Fallback selector',
    useSelectedModel: 'Use selected model',
    switchWarning: 'Switching checkpoint needs a fresh enrollment profile. Choose rebuild if audio samples are still in this session.',
    architecture: 'Architecture',
    evaluationResults: 'Evaluation Results',
    loading: 'Loading...',
    noKeywordsEnrolled: 'No keywords enrolled yet',
    noDemoCheckpoints: 'No demo checkpoints found',
    noModelProfiles: 'No model profiles found',
    noActiveProfile: 'No active profile',
    noActiveModel: 'No active model',
    readyLower: 'ready',
    missingCheckpoint: 'missing checkpoint',
    active: 'ACTIVE',
    missing: 'MISSING',
    readyUpper: 'READY',
    thrAuto: 'thr auto',
    thr: 'thr',
    loadedFile: 'Loaded: {name}',
    fileLoaded: 'File loaded',
    recorded: 'Recorded',
    recording: 'Recording...',
    audioRecorded: 'Audio recorded',
    micDenied: 'Microphone access denied',
    enterKeywordName: 'Enter keyword name first',
    failed: 'Failed',
    networkError: 'Network error',
    enterKeywords: 'Enter keywords',
    enrollingGsc: 'Enrolling from GSC...',
    enrolling: 'Enrolling...',
    addedLog: 'Added "{word}" ({count} samples, thr={threshold})',
    enrolledSample: 'Enrolled sample for "{word}"',
    enrolledCount: 'Enrolled {count} keywords',
    clearConfirm: 'Clear all enrolled keywords?',
    allKeywordsCleared: 'All keywords cleared',
    clearedAllKeywords: 'Cleared all keywords',
    profileSaved: 'Profile "{name}" saved',
    profileLoaded: 'Loaded "{name}" ({count} kw)',
    savedProfiles: 'Saved: {profiles}',
    noProfiles: 'No profiles yet',
    uploadOrRecordFirst: 'Upload or record audio first',
    detecting: 'Detecting...',
    detected: 'Detected',
    rejected: 'Rejected',
    unknown: 'UNKNOWN',
    confidence: 'confidence',
    top2Margin: 'top-2 margin',
    topCandidates: 'top candidates',
    dist: 'dist',
    uploadAudioFirst: 'Upload audio first',
    analyzing: 'Analyzing...',
    results: 'Results',
    sequence: 'Sequence:',
    predictedSequence: 'Predicted sequence:',
    expectedSequence: 'Expected sequence:',
    noKeywordsDetected: 'No keywords detected',
    expectedMismatch: 'Expected label count ({expected}) does not match returned {unit} ({actual}). The table still shows Expected by file order, but total accuracy is not computed because VAD/cooldown may have skipped words.',
    expectedNotCompared: 'Expected labels not compared:',
    extraDetections: 'Extra detections beyond the label file:',
    duration: 'Duration',
    expectedCount: 'Expected',
    detectionCount: 'Detected',
    matched: 'Matched',
    allAccuracy: 'All accuracy',
    enrolledOnlyAccuracy: 'Enrolled-only accuracy',
    outOfEnrollment: 'Out of enrollment',
    noOutOfEnrollment: 'All expected labels are enrolled',
    outOfEnrollmentWarning: '{count} expected labels are not enrolled: {labels}. All accuracy includes them; enrolled-only accuracy ignores them.',
    noExpectedOverlap: 'NO_OVERLAP',
    missedExpected: 'MISS',
    missedExpectedLabels: 'Missed expected labels',
    noDetectionForExpected: 'No detection overlapped this expected word',
    missReason: 'Why it missed',
    missReasonNoOverlap: 'No detection overlapped this expected word. VAD, segmentation, or cooldown likely skipped it.',
    missReasonWrongPrediction: 'A detection overlapped this expected word, but predicted "{predicted}" instead.',
    missReasonRejected: 'A segment overlapped this expected word but was rejected by threshold.',
    missReasonLowMargin: 'A segment overlapped this expected word and top-1 was "{predicted}", but top-2 was too close. Margin {margin} is below the accept margin.',
    missReasonRejectedGeneric: 'A segment overlapped this expected word but was rejected by the active detector policy.',
    overlappedDetection: 'Overlapped detection',
    margin: 'Margin',
    time: 'Time',
    predicted: 'Predicted',
    expected: 'Expected',
    match: 'Match',
    top3Candidates: 'Top 3 Candidates',
    l2Dist: 'L2 Dist',
    status: 'Status',
    uploadGroundTruthFirst: 'Upload a TXT ground truth file first',
    evaluatedFiles: 'Evaluated {total} files - {accuracy}% accuracy',
    notFound: 'Not Found',
    wrong: 'Wrong',
    modelProfileNotFound: 'Model profile not found',
    checkpointMissing: 'Checkpoint file is missing',
    modelAlreadyActive: 'This model is already active',
    noSessionAudioToRebuild: 'No session audio to rebuild',
    rebuildKeepsSamples: 'Rebuild keeps current session audio samples and recomputes prototypes with the new checkpoint.',
    clearOnlyValid: 'No session audio samples are available, so clearing enrollment is the valid option.',
    chooseModelFirst: 'Choose a model profile first',
    couldNotSwitchModel: 'Could not switch model',
    switchNetworkError: 'Network error while switching model',
    switchedModelLogRebuilt: 'Switched model to {model} and rebuilt enrollment',
    switchedModelLogCleared: 'Switched model to {model} and cleared enrollment',
    switchedRebuilt: 'Model switched; enrollment rebuilt',
    switchedCleared: 'Model switched; enrollment cleared',
    couldNotLoadProfiles: 'Could not load model profiles from the API.',
    profileApiFailed: 'Model profile API failed',
    noEvalData: 'No evaluation data',
    noEvalResults: 'No evaluation results found',
    couldNotLoad: 'Could not load',
    chooseModelProfile: 'Choose a model profile',
    modelLabels: {
      Profile: 'Profile',
      Architecture: 'Architecture',
      Parameters: 'Parameters',
      Embedding: 'Embedding',
      Feature: 'Feature',
      Device: 'Device',
      Input: 'Input',
      Checkpoint: 'Checkpoint',
    },
  },
  vi: {
    skipMain: 'Bỏ qua đến nội dung chính',
    switchCheckpoint: 'Đổi checkpoint',
    chooseEnrollmentHandling: 'Chọn cách xử lý enrollment',
    closeModelSwitchDialog: 'Đóng hộp thoại đổi model',
    switchRebuild: 'Đổi và dựng lại enrollment',
    switchClear: 'Đổi và xóa enrollment',
    cancel: 'Hủy',
    primaryNavigation: 'Điều hướng chính',
    sectionNavigation: 'Điều hướng khu vực',
    mobileAppHeader: 'Thanh đầu trang mobile',
    brandSub: 'Bảng điều khiển nghiên cứu',
    mobileBrandSub: 'Phòng thử keyword',
    main: 'Chính',
    advanced: 'Nâng cao',
    enrollment: 'Ghi danh',
    detection: 'Nhận diện',
    detect: 'Nhận diện',
    longFile: 'File dài',
    streaming: 'Streaming',
    openSetTest: 'Kiểm thử open-set',
    openSet: 'Open-set',
    modelInfo: 'Thông tin model',
    info: 'Thông tin',
    modelLoaded: 'Model đã tải',
    sidebarCaption: 'DSCNN-L - Prototypical Net',
    ready: 'Sẵn sàng',
    activeCheckpoint: 'Checkpoint đang dùng',
    loadingModel: 'Đang tải model...',
    reloadProfiles: 'Tải lại profile',
    loadingCheckpoints: 'Đang tải checkpoint...',
    modelCheckpointSwitcher: 'Bộ đổi checkpoint model',
    kwsConsole: 'Bảng điều khiển KWS',
    commandCenter: 'Trung tâm điều khiển keyword',
    heroText: 'Ghi danh một tập mẫu nhỏ, hiệu chỉnh ngưỡng open-set và kiểm thử nhận diện keyword tĩnh hoặc streaming trong cùng một workspace.',
    fewShotEnrollment: 'Ghi danh few-shot',
    openSetRejection: 'Từ chối open-set',
    streamingTest: 'Kiểm thử streaming',
    prototypeCalibration: 'Hiệu chỉnh prototype',
    bestCheckpoint: 'Checkpoint tốt nhất',
    gscDataset: 'Bộ dữ liệu GSC',
    quickPreset: 'Preset nhanh',
    keywords: 'Từ khóa',
    enrollFromGsc: 'Ghi danh từ GSC',
    microphone: 'Micro',
    keywordName: 'Tên keyword',
    clickToRecord: 'Bấm để thu khoảng 1s',
    uploadFile: 'Tải file lên',
    profiles: 'Profile',
    profileName: 'Tên profile',
    save: 'Lưu',
    load: 'Tải',
    enrolledKeywords: 'Keyword đã ghi danh',
    clearAll: 'Xóa tất cả',
    activityLog: 'Nhật ký thao tác',
    actionsHere: 'Thao tác sẽ hiển thị ở đây',
    detectionDesc: 'Tải lên hoặc thu khoảng 1 giây audio để nhận diện keyword đã ghi danh',
    audioInput: 'Audio đầu vào',
    chooseDetectionAudio: 'Chọn file audio để nhận diện',
    dropAudio: 'Thả file audio vào đây hoặc bấm để chọn',
    dropAudioSub: 'WAV, MP3, OGG - khoảng 1 giây',
    recordDetectionSample: 'Thu mẫu audio để nhận diện',
    orRecordWithMic: 'Hoặc thu bằng micro',
    settings: 'Thiết lập',
    detectionThreshold: 'Ngưỡng nhận diện',
    usePerClassThreshold: 'Dùng ngưỡng từng lớp',
    closeWordGuard: 'Chặn từ gần nhau',
    closeWordGuardHelp: 'Reject các kết quả chưa chắc chắn khi top-1 và top-2 quá sát nhau.',
    closeWordGuardOn: 'Guard BẬT',
    closeWordGuardOff: 'Guard TẮT',
    acceptMargin: 'Margin accept',
    perClassOn: 'Ngưỡng lớp BẬT',
    perClassOff: 'Ngưỡng lớp TẮT',
    detectKeyword: 'Nhận diện keyword',
    uploadThenDetect: 'Tải lên hoặc thu audio, sau đó bấm Nhận diện',
    distanceToPrototypes: 'Khoảng cách tới prototype',
    mfccFeatures: 'Đặc trưng MFCC',
    detectionHistory: 'Lịch sử nhận diện',
    clear: 'Xóa',
    noDetectionsYet: 'Chưa có kết quả nhận diện',
    longFileDetection: 'Nhận diện file dài',
    longFileDesc: 'Tải lên file audio dài hơn (5-30s) để nhận diện nhiều từ',
    audioFile: 'File audio',
    chooseLongAudio: 'Chọn file audio dài',
    dropLongAudio: 'Thả file audio dài (5-30 giây)',
    threshold: 'Ngưỡng',
    segmentationMethod: 'Cách tách đoạn',
    energyBased: 'Dựa trên năng lượng',
    sileroVad: 'Silero VAD',
    minDuration: 'Thời lượng tối thiểu (ms)',
    perClassThreshold: 'Ngưỡng từng lớp',
    distanceMetric: 'Metric khoảng cách',
    detectWords: 'Nhận diện từ',
    groundTruth: 'Nhãn đúng',
    optional: 'Tùy chọn',
    gtHelp: 'Tải file TXT để kiểm tra accuracy sau khi nhận diện',
    chooseGroundTruth: 'Chọn file nhãn đúng',
    dropTxt: 'Thả file .txt vào đây',
    gtFormat: 'Định dạng: expected_keyword (mỗi dòng một nhãn, đúng thứ tự segment)',
    txtFormat: 'Định dạng TXT:',
    gtFormatHelp: 'Có thể dùng dấu phẩy hoặc mỗi dòng một nhãn. Thứ tự khớp với các segment đã nhận diện.',
    labelsInFile: 'Label trong file',
    labelCount: '{count} label',
    labelsLoaded: 'Đã tải {count} label từ {name}',
    moreLabels: '+{count} label nữa',
    invalidGroundTruth: 'Không tìm thấy label trong file Ground Truth',
    timingJson: 'Timing JSON',
    timingHelp: 'Tùy chọn: tải .timings.json để so sánh detection theo overlap thời gian',
    chooseTimingJson: 'Chọn file timing JSON',
    dropTimingJson: 'Thả .timings.json vào đây',
    timingsInFile: 'Timing trong file',
    timingCount: '{count} mốc',
    timingsLoaded: 'Đã tải {count} mốc timing từ {name}',
    invalidTimingJson: 'Không tìm thấy timing hợp lệ trong file JSON',
    timingMatchMode: 'Chế độ timing: Expected được chọn theo overlap thời gian lớn nhất với mỗi detection.',
    expectedTimeline: 'Timing nhãn đúng',
    detectedTimeline: 'Timing nhận diện',
    segments: 'Đoạn',
    correct: 'Đúng',
    accuracy: 'Accuracy',
    uploadLongToAnalyze: 'Tải lên file audio dài để phân tích',
    streamingTitle: 'Streaming thời gian thực',
    streamingDesc: 'Nhận diện keyword trực tiếp từ micro bằng cửa sổ trượt',
    startStreaming: 'Bắt đầu streaming',
    stopStreaming: 'Dừng streaming',
    listening: 'Đang nghe...',
    stopped: 'Đã dừng',
    streamingStarted: 'Đã bắt đầu streaming',
    streamingStopped: 'Đã dừng streaming - {count} lượt nhận diện',
    webSocketError: 'Lỗi WebSocket',
    liveStats: 'Thống kê trực tiếp',
    detections: 'Lượt nhận diện',
    lastKeyword: 'Keyword gần nhất',
    elapsed: 'Thời gian',
    detectionFeed: 'Luồng nhận diện',
    startStreamingEmpty: 'Bắt đầu streaming để xem kết quả trực tiếp',
    openSetTitle: 'Kiểm thử từ chối open-set',
    openSetDesc: 'Kiểm tra các từ chưa ghi danh có được từ chối đúng là "unknown" hay không',
    unknownWords: 'Các từ GSC unknown để test',
    samplesPerWord: 'Số mẫu mỗi từ',
    runTest: 'Chạy test',
    runOpenSetTest: 'Chạy kiểm thử open-set',
    randomSeed: 'Seed ngẫu nhiên',
    openSetInputHelp: 'Từ known lấy từ các keyword đang enroll; từ unknown được sample từ thư mục GSC local.',
    runningOpenSet: 'Đang chạy kiểm thử open-set...',
    knownTested: 'Known đã test',
    unknownTested: 'Unknown đã test',
    openSetAcc: 'Open-set ACC',
    keywordAcc: 'Keyword ACC',
    unknownRejectAcc: 'Unknown reject ACC',
    falseAcceptRate: 'False accept rate',
    falseRejectRate: 'False reject rate',
    falseAccepts: 'False accepts',
    knownMisses: 'Known misses',
    noFalseAccepts: 'Không có false accept',
    noKnownMisses: 'Không có known miss',
    sourceWord: 'Từ nguồn',
    file: 'File',
    skippedUnknownWords: 'Unknown bị bỏ qua',
    missingKnownWords: 'Known thiếu audio',
    missingUnknownWords: 'Unknown thiếu audio',
    shortAudioWords: 'Không đủ mẫu',
    openSetEvaluated: 'Đã chạy open-set: {accuracy} open-set ACC',
    openSetPreset: 'Preset open-set',
    manualSplit: 'Split tu nhap',
    knownWords: 'Tu GSC known/da enroll',
    openSetSplitSummary: 'Known: {known} tu. Unknown: {unknown} tu. Holdout: {heldout}.',
    calibrateOpenSet: 'Hieu chinh open-set',
    runningCalibration: 'Dang hieu chinh...',
    candidateLabels: 'Candidate label',
    balancedScore: 'Balanced score',
    openSetCalibrated: 'Da hieu chinh: {score} balanced score',
    calibrationResults: 'Ket qua calibration',
    bestBalanced: 'Can bang tot nhat',
    bestRejectUnknown: 'Reject unknown tot nhat',
    bestRecognizeKeyword: 'Nhan keyword tot nhat',
    applySettings: 'Ap dung thiet lap',
    settingsApplied: 'Da ap dung thiet lap calibration',
    thresholdValue: 'Nguong',
    perClass: 'Nguong tung lop',
    policy: 'Policy',
    language: 'Ngôn ngữ',
    comingSoon: 'Sắp có - cần dữ liệu GSC',
    resultsHere: 'Kết quả sẽ hiển thị ở đây',
    modelInformation: 'Thông tin model',
    modelInformationDesc: 'Chi tiết kiến trúc và kết quả đánh giá',
    modelCheckpoint: 'Checkpoint model',
    fallbackSelector: 'Bộ chọn dự phòng',
    useSelectedModel: 'Dùng model đã chọn',
    switchWarning: 'Đổi checkpoint cần profile enrollment mới. Chọn dựng lại nếu audio mẫu còn trong phiên này.',
    architecture: 'Kiến trúc',
    evaluationResults: 'Kết quả đánh giá',
    loading: 'Đang tải...',
    noKeywordsEnrolled: 'Chưa ghi danh keyword nào',
    noDemoCheckpoints: 'Không tìm thấy checkpoint demo',
    noModelProfiles: 'Không tìm thấy profile model',
    noActiveProfile: 'Chưa có profile đang dùng',
    noActiveModel: 'Chưa có model đang dùng',
    readyLower: 'sẵn sàng',
    missingCheckpoint: 'thiếu checkpoint',
    active: 'ĐANG DÙNG',
    missing: 'THIẾU',
    readyUpper: 'SẴN SÀNG',
    thrAuto: 'ngưỡng auto',
    thr: 'ngưỡng',
    loadedFile: 'Đã tải: {name}',
    fileLoaded: 'Đã tải file',
    recorded: 'Đã thu',
    recording: 'Đang thu...',
    audioRecorded: 'Đã thu audio',
    micDenied: 'Không được cấp quyền micro',
    enterKeywordName: 'Nhập tên keyword trước',
    failed: 'Thất bại',
    networkError: 'Lỗi mạng',
    enterKeywords: 'Nhập keyword',
    enrollingGsc: 'Đang ghi danh từ GSC...',
    enrolling: 'Đang ghi danh...',
    addedLog: 'Đã thêm "{word}" ({count} mẫu, ngưỡng={threshold})',
    enrolledSample: 'Đã ghi danh mẫu cho "{word}"',
    enrolledCount: 'Đã ghi danh {count} keyword',
    clearConfirm: 'Xóa toàn bộ keyword đã ghi danh?',
    allKeywordsCleared: 'Đã xóa toàn bộ keyword',
    clearedAllKeywords: 'Đã xóa toàn bộ keyword',
    profileSaved: 'Đã lưu profile "{name}"',
    profileLoaded: 'Đã tải "{name}" ({count} kw)',
    savedProfiles: 'Đã lưu: {profiles}',
    noProfiles: 'Chưa có profile',
    uploadOrRecordFirst: 'Tải lên hoặc thu audio trước',
    detecting: 'Đang nhận diện...',
    detected: 'Đã phát hiện',
    rejected: 'Bị từ chối',
    unknown: 'UNKNOWN',
    confidence: 'độ tin cậy',
    top2Margin: 'margin top-2',
    topCandidates: 'ứng viên gần nhất',
    dist: 'kc',
    uploadAudioFirst: 'Tải audio lên trước',
    analyzing: 'Đang phân tích...',
    results: 'Kết quả',
    sequence: 'Chuỗi:',
    predictedSequence: 'Chuỗi dự đoán:',
    expectedSequence: 'Chuỗi nhãn đúng:',
    noKeywordsDetected: 'Không phát hiện keyword',
    expectedMismatch: 'Số nhãn đúng ({expected}) không khớp số {unit} trả về ({actual}). Bảng vẫn hiển thị Expected theo thứ tự trong file, nhưng không tính accuracy tổng vì VAD/cooldown có thể đã bỏ qua từ.',
    expectedNotCompared: 'Nhãn đúng chưa được so sánh:',
    extraDetections: 'Detection vượt quá số nhãn trong file:',
    duration: 'Thời lượng',
    expectedCount: 'Nhãn đúng',
    detectionCount: 'Đã detect',
    matched: 'Khớp',
    allAccuracy: 'Accuracy tổng',
    enrolledOnlyAccuracy: 'Accuracy keyword đã enroll',
    outOfEnrollment: 'Ngoài enrollment',
    noOutOfEnrollment: 'Tất cả nhãn đúng đã được enroll',
    outOfEnrollmentWarning: '{count} nhãn đúng chưa được enroll: {labels}. Accuracy tổng vẫn tính chúng; accuracy keyword đã enroll sẽ bỏ qua chúng.',
    noExpectedOverlap: 'KHÔNG_OVERLAP',
    missedExpected: 'MISS',
    missedExpectedLabels: 'Nhãn đúng bị miss',
    noDetectionForExpected: 'Không có detection nào overlap với từ đúng này',
    missReason: 'Lý do miss',
    missReasonNoOverlap: 'Không có detection nào overlap với từ đúng này. VAD, tách đoạn hoặc cooldown có thể đã bỏ qua.',
    missReasonWrongPrediction: 'Có detection overlap với từ đúng này, nhưng model dự đoán là "{predicted}".',
    missReasonRejected: 'Có segment overlap với từ đúng này nhưng bị reject bởi threshold.',
    missReasonLowMargin: 'Có segment overlap với từ đúng này và top-1 là "{predicted}", nhưng top-2 quá sát. Margin {margin} thấp hơn ngưỡng accept.',
    missReasonRejectedGeneric: 'Có segment overlap với từ đúng này nhưng bị reject bởi policy đang bật.',
    overlappedDetection: 'Detection overlap',
    margin: 'Margin',
    time: 'Thời gian',
    predicted: 'Dự đoán',
    expected: 'Nhãn đúng',
    match: 'Khớp',
    top3Candidates: 'Top 3 ứng viên',
    l2Dist: 'Khoảng cách L2',
    status: 'Trạng thái',
    uploadGroundTruthFirst: 'Tải file TXT nhãn đúng trước',
    evaluatedFiles: 'Đã đánh giá {total} file - accuracy {accuracy}%',
    notFound: 'Không tìm thấy',
    wrong: 'Sai',
    modelProfileNotFound: 'Không tìm thấy profile model',
    checkpointMissing: 'Thiếu file checkpoint',
    modelAlreadyActive: 'Model này đang được dùng',
    noSessionAudioToRebuild: 'Không có audio trong phiên để dựng lại',
    rebuildKeepsSamples: 'Dựng lại sẽ giữ audio mẫu trong phiên hiện tại và tính lại prototype bằng checkpoint mới.',
    clearOnlyValid: 'Không có audio mẫu trong phiên, vì vậy chỉ có thể xóa enrollment.',
    chooseModelFirst: 'Chọn một profile model trước',
    couldNotSwitchModel: 'Không đổi được model',
    switchNetworkError: 'Lỗi mạng khi đổi model',
    switchedModelLogRebuilt: 'Đã đổi model sang {model} và dựng lại enrollment',
    switchedModelLogCleared: 'Đã đổi model sang {model} và xóa enrollment',
    switchedRebuilt: 'Đã đổi model; enrollment được dựng lại',
    switchedCleared: 'Đã đổi model; enrollment đã xóa',
    couldNotLoadProfiles: 'Không tải được profile model từ API.',
    profileApiFailed: 'API profile model lỗi',
    noEvalData: 'Không có dữ liệu đánh giá',
    noEvalResults: 'Không tìm thấy kết quả đánh giá',
    couldNotLoad: 'Không tải được',
    chooseModelProfile: 'Chọn profile model',
    modelLabels: {
      Profile: 'Profile',
      Architecture: 'Kiến trúc',
      Parameters: 'Tham số',
      Embedding: 'Embedding',
      Feature: 'Đặc trưng',
      Device: 'Thiết bị',
      Input: 'Đầu vào',
      Checkpoint: 'Checkpoint',
    },
  },
};

const STATIC_TEXT_KEYS = [
  'skipMain', 'switchCheckpoint', 'chooseEnrollmentHandling', 'switchRebuild', 'switchClear', 'cancel',
  'brandSub', 'mobileBrandSub', 'main', 'advanced', 'enrollment', 'detection', 'detect', 'longFile',
  'streaming', 'openSetTest', 'openSet', 'modelInfo', 'info', 'modelLoaded', 'sidebarCaption', 'ready',
  'activeCheckpoint', 'loadingModel', 'reloadProfiles', 'loadingCheckpoints', 'kwsConsole', 'commandCenter',
  'heroText', 'fewShotEnrollment', 'openSetRejection', 'streamingTest', 'prototypeCalibration', 'bestCheckpoint',
  'gscDataset', 'quickPreset', 'keywords', 'enrollFromGsc', 'microphone', 'keywordName', 'clickToRecord',
  'uploadFile', 'profiles', 'profileName', 'save', 'load', 'enrolledKeywords', 'clearAll', 'activityLog',
  'actionsHere', 'detectionDesc', 'audioInput', 'dropAudio', 'dropAudioSub', 'orRecordWithMic', 'settings',
  'detectionThreshold', 'usePerClassThreshold', 'closeWordGuard', 'closeWordGuardHelp', 'closeWordGuardOn', 'closeWordGuardOff', 'acceptMargin', 'perClassOn', 'perClassOff', 'detectKeyword', 'uploadThenDetect', 'distanceToPrototypes',
  'mfccFeatures', 'detectionHistory', 'clear', 'noDetectionsYet', 'longFileDetection', 'longFileDesc',
  'audioFile', 'dropLongAudio', 'threshold', 'segmentationMethod', 'energyBased', 'sileroVad', 'minDuration',
  'perClassThreshold', 'distanceMetric', 'detectWords', 'groundTruth', 'optional', 'gtHelp', 'dropTxt',
  'gtFormat', 'txtFormat', 'gtFormatHelp', 'dropTimingJson', 'timingHelp', 'segments', 'correct', 'accuracy', 'uploadLongToAnalyze',
  'streamingTitle', 'streamingDesc', 'startStreaming', 'liveStats', 'detections', 'lastKeyword', 'elapsed',
  'detectionFeed', 'startStreamingEmpty', 'openSetTitle', 'openSetDesc', 'unknownWords', 'samplesPerWord',
  'runTest', 'runOpenSetTest', 'randomSeed', 'openSetInputHelp', 'resultsHere', 'modelInformation', 'modelInformationDesc', 'modelCheckpoint', 'fallbackSelector',
  'useSelectedModel', 'switchWarning', 'architecture', 'evaluationResults', 'loading',
];

function formatText(template, vars = {}) {
  return String(template ?? '').replace(/\{(\w+)\}/g, (_, key) => vars[key] ?? '');
}

function t(key, vars = {}) {
  const langPack = I18N[currentLang] || I18N.en;
  const value = langPack[key] ?? I18N.en[key] ?? key;
  return typeof value === 'string' ? formatText(value, vars) : value;
}

function profileText(profile, field) {
  if (!profile) return '';
  return profile[`${field}_${currentLang}`] || profile[field] || profile[`${field}_en`] || profile[`${field}_vi`] || '';
}

function buildStaticTextLookup() {
  const lookup = new Map();
  for (const key of STATIC_TEXT_KEYS) {
    for (const lang of Object.keys(I18N)) {
      const value = I18N[lang][key];
      if (typeof value === 'string') lookup.set(value, key);
    }
  }
  return lookup;
}

const STATIC_TEXT_LOOKUP = buildStaticTextLookup();

function localizeStaticTextNodes(root = document.body) {
  if (!root) return;
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      const parent = node.parentElement;
      if (!parent || ['SCRIPT', 'STYLE', 'TEXTAREA'].includes(parent.tagName)) {
        return NodeFilter.FILTER_REJECT;
      }
      return node.nodeValue.trim() ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
    },
  });
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  for (const node of nodes) {
    const raw = node.nodeValue;
    const trimmed = raw.trim();
    const key = STATIC_TEXT_LOOKUP.get(trimmed);
    if (!key) continue;
    const leading = raw.match(/^\s*/)?.[0] || '';
    const trailing = raw.match(/\s*$/)?.[0] || '';
    node.nodeValue = leading + t(key) + trailing;
  }
}

function setText(id, key) {
  const el = document.getElementById(id);
  if (el) el.textContent = t(key);
}

function applyAttributes() {
  const attrs = [
    ['modelSwitchCloseBtn', 'aria-label', 'closeModelSwitchDialog'],
    ['micBtn', 'aria-label', 'recordDetectionSample'],
    ['detectMicBtn', 'aria-label', 'recordDetectionSample'],
    ['detectDrop', 'aria-label', 'chooseDetectionAudio'],
    ['longDrop', 'aria-label', 'chooseLongAudio'],
    ['gtDrop', 'aria-label', 'chooseGroundTruth'],
    ['timingDrop', 'aria-label', 'chooseTimingJson'],
  ];
  attrs.forEach(([id, attr, key]) => {
    const el = document.getElementById(id);
    if (el) el.setAttribute(attr, t(key));
  });
  const micKeyword = document.getElementById('micKeyword');
  if (micKeyword) micKeyword.placeholder = currentLang === 'vi' ? 'vd: xin_chao, hey_jarvis' : 'e.g. hello, hey_jarvis';
  document.querySelector('.sidebar')?.setAttribute('aria-label', t('primaryNavigation'));
  document.querySelector('.mobile-topbar')?.setAttribute('aria-label', t('mobileAppHeader'));
  document.querySelector('.mobile-nav')?.setAttribute('aria-label', t('sectionNavigation'));
  document.querySelector('.model-switcher-shell')?.setAttribute('aria-label', t('modelCheckpointSwitcher'));
  document.querySelector('.language-toggle')?.setAttribute('aria-label', t('language'));
}

function updateLanguageButtons() {
  document.querySelectorAll('[data-lang]').forEach(btn => {
    const active = btn.dataset.lang === currentLang;
    btn.classList.toggle('active', active);
    btn.setAttribute('aria-pressed', String(active));
  });
}

function applyLanguage(lang = currentLang) {
  currentLang = I18N[lang] ? lang : 'en';
  localStorage.setItem('kws_lang', currentLang);
  document.documentElement.lang = currentLang;
  localizeStaticTextNodes();
  applyAttributes();
  updateLanguageButtons();
  renderModelProfiles();
  renderGroundTruthPreview();
  renderDetectionHistory();
  updateOpenSetSplitPreview();
  if (lastModelInfo) renderModelInfo(lastModelInfo);
}

function setLanguage(lang) {
  applyLanguage(lang);
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, char => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  }[char]));
}

function metricText(value, digits = 3) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(digits) : '-';
}

function checkboxFormValue(id) {
  return document.getElementById(id)?.checked ? 'true' : 'false';
}

function setBusy(id, busy, label) {
  const btn = document.getElementById(id);
  if (!btn) return;
  if (busy) {
    btn.dataset.originalHtml = btn.innerHTML;
    btn.disabled = true;
    btn.classList.add('is-busy');
    if (label) btn.innerHTML = `<span class="btn-loader" aria-hidden="true"></span>${label}`;
    return;
  }
  btn.disabled = false;
  btn.classList.remove('is-busy');
  if (btn.dataset.originalHtml) {
    btn.innerHTML = btn.dataset.originalHtml;
    delete btn.dataset.originalHtml;
  }
}

// Navigation
const tabButtons = Array.from(document.querySelectorAll('.nav-item[data-tab]'));

function setActiveTab(tab) {
  tabButtons.forEach(btn => {
    const isActive = btn.dataset.tab === tab;
    btn.classList.toggle('active', isActive);
    btn.setAttribute('role', 'tab');
    btn.setAttribute('aria-selected', String(isActive));
    btn.setAttribute('aria-controls', 'tab-' + btn.dataset.tab);
    btn.setAttribute('type', 'button');
  });
  document.querySelectorAll('.tab-panel').forEach(panel => {
    const isActive = panel.id === 'tab-' + tab;
    panel.classList.toggle('active', isActive);
    panel.hidden = !isActive;
    panel.setAttribute('role', 'tabpanel');
    if (isActive) {
      const activeButton = tabButtons.find(btn => btn.dataset.tab === tab);
      panel.setAttribute('aria-label', activeButton?.textContent.trim() || tab);
    }
  });
}

tabButtons.forEach(btn => {
  btn.addEventListener('click', () => {
    setActiveTab(btn.dataset.tab);
  });
  btn.addEventListener('keydown', event => {
    if (!['ArrowRight', 'ArrowLeft'].includes(event.key)) return;
    event.preventDefault();
    const group = Array.from(btn.closest('nav, aside')?.querySelectorAll('.nav-item[data-tab]') || tabButtons);
    const delta = event.key === 'ArrowRight' ? 1 : -1;
    const next = group[(group.indexOf(btn) + delta + group.length) % group.length];
    next.focus();
    setActiveTab(next.dataset.tab);
  });
});
setActiveTab(document.querySelector('.nav-item.active')?.dataset.tab || 'enroll');

// Toast
function toast(type, msg) {
  const c = document.getElementById('toasts');
  const t = document.createElement('div');
  t.className = 'toast ' + type;
  t.setAttribute('role', type === 'error' ? 'alert' : 'status');
  t.textContent = msg;
  c.appendChild(t);
  setTimeout(() => { t.style.opacity = '0'; t.style.transition = 'opacity .18s'; setTimeout(() => t.remove(), 180); }, 3000);
}

// Drag and drop
function setupDrop(zoneId, fileId, target) {
  const zone = document.getElementById(zoneId);
  const input = document.getElementById(fileId);
  if (!zone || !input) return;
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', e => {
    e.preventDefault(); zone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
      audioBlobs[target] = e.dataTransfer.files[0];
      zone.querySelector('p').textContent = t('loadedFile', { name: e.dataTransfer.files[0].name });
      toast('success', t('fileLoaded'));
    }
  });
  input.addEventListener('change', () => {
    if (input.files.length) {
      audioBlobs[target] = input.files[0];
      zone.querySelector('p').textContent = t('loadedFile', { name: input.files[0].name });
      toast('success', t('fileLoaded'));
    }
  });
}

// Microphone
async function toggleMicRecord(target) {
  if (mediaRecorder && mediaRecorder.state === 'recording') { mediaRecorder.stop(); return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioCtx = new AudioContext({ sampleRate: 16000 });
    const source = audioCtx.createMediaStreamSource(stream);
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 2048;
    source.connect(analyser);
    const chunks = [];
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = e => chunks.push(e.data);
    mediaRecorder.onstop = () => {
      stream.getTracks().forEach(t => t.stop());
      cancelAnimationFrame(recAnimFrame);
      audioBlobs[target] = new Blob(chunks, { type: 'audio/wav' });
      const btnId = target === 'enroll' ? 'micBtn' : 'detectMicBtn';
      document.getElementById(btnId).classList.remove('recording');
      const statusId = target === 'enroll' ? 'micStatus' : 'detectMicStatus';
      document.getElementById(statusId).textContent = t('recorded');
      toast('success', t('audioRecorded'));
      if (target === 'enroll') autoEnrollMic();
    };
    mediaRecorder.start();
    const btnId = target === 'enroll' ? 'micBtn' : 'detectMicBtn';
    document.getElementById(btnId).classList.add('recording');
    const statusId = target === 'enroll' ? 'micStatus' : 'detectMicStatus';
    document.getElementById(statusId).textContent = t('recording');
    drawWaveform(target);
    setTimeout(() => { if (mediaRecorder?.state === 'recording') mediaRecorder.stop(); }, 1500);
  } catch { toast('error', t('micDenied')); }
}

function drawWaveform(target) {
  const canvasId = target === 'enroll' ? 'enrollWaveform' : 'detectWaveform';
  const canvas = document.getElementById(canvasId);
  if (!canvas || !analyser) return;
  const ctx = canvas.getContext('2d');
  canvas.width = canvas.offsetWidth * 2;
  canvas.height = canvas.offsetHeight * 2;
  const w = canvas.width, h = canvas.height;
  const data = new Uint8Array(analyser.frequencyBinCount);
  function draw() {
    recAnimFrame = requestAnimationFrame(draw);
    analyser.getByteTimeDomainData(data);
    ctx.fillStyle = cssVar('--bg-input') || '#f8fafc';
    ctx.fillRect(0, 0, w, h);
    ctx.lineWidth = 2;
    ctx.strokeStyle = cssVar('--accent-500') || '#0f766e';
    ctx.beginPath();
    const sliceW = w / data.length;
    for (let i = 0; i < data.length; i++) {
      const y = (data[i] / 128.0) * h / 2;
      i === 0 ? ctx.moveTo(0, y) : ctx.lineTo(i * sliceW, y);
    }
    ctx.stroke();
  }
  draw();
}

// Auto-enroll mic
async function autoEnrollMic() {
  const keyword = document.getElementById('micKeyword').value.trim();
  if (!keyword) { toast('error', t('enterKeywordName')); return; }
  if (!audioBlobs.enroll) return;
  const fd = new FormData();
  fd.append('keyword', keyword);
  fd.append('audio', audioBlobs.enroll, 'rec.wav');
  try {
    const r = await fetch(API + '/api/enroll/mic', { method: 'POST', body: fd });
    const d = await r.json();
    if (r.ok) {
      addLog(t('addedLog', { word: d.word, count: d.count, threshold: d.threshold }));
      toast('success', t('enrolledSample', { word: d.word }));
      refreshEnrolled();
    } else toast('error', d.error || t('failed'));
  } catch { toast('error', t('networkError')); }
}

// Enroll GSC
async function enrollGSC() {
  const words = document.getElementById('gscWords').value.trim();
  if (!words) { toast('error', t('enterKeywords')); return; }
  toast('info', t('enrollingGsc'));
  setBusy('enrollGscBtn', true, t('enrolling'));
  const fd = new FormData();
  fd.append('words', words);
  fd.append('k', '5');
  try {
    const r = await fetch(API + '/api/enroll/gsc', { method: 'POST', body: fd });
    const d = await r.json();
    if (d.results) {
      d.results.forEach(res => {
        addLog(res.status === 'ok'
          ? `[OK] "${res.word}" (${res.samples} samples, ${t('thr')}=${res.threshold})`
          : `[ERR] "${res.word}": ${res.status}`);
      });
      toast('success', t('enrolledCount', { count: d.enrolled }));
      refreshEnrolled();
    }
  } catch {
    toast('error', t('networkError'));
  } finally {
    setBusy('enrollGscBtn', false);
  }
}

// Clear
async function clearAll() {
  if (!window.confirm(t('clearConfirm'))) return;
  await fetch(API + '/api/enroll/clear', { method: 'POST' });
  toast('info', t('allKeywordsCleared'));
  addLog(t('clearedAllKeywords'));
  refreshEnrolled();
}

// Refresh enrolled
async function refreshEnrolled() {
  try {
    const r = await fetch(API + '/api/enroll/status');
    const d = await r.json();
    const c = document.getElementById('enrolledChips');
    if (d.total === 0) {
      enrolledKeywordsState = new Set();
      c.innerHTML = `<div class="empty-state compact">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/></svg>
        <p>${escapeHtml(t('noKeywordsEnrolled'))}</p>
      </div>`;
      return;
    }
    enrolledKeywordsState = new Set(Object.keys(d.enrolled || {}).map(w => String(w).toLowerCase()));
    c.innerHTML = Object.entries(d.enrolled).map(([w, info]) =>
      `<div class="chip chip-rich">
        <span>${escapeHtml(w)}</span>
        <span class="chip-count">${escapeHtml(info.count)}</span>
        <span class="chip-detail">${escapeHtml(t('thr'))} ${escapeHtml(info.threshold)} &middot; ${escapeHtml(info.profile)}</span>
      </div>`
    ).join('');
  } catch {}
}

// Log
function addLog(msg) {
  const el = document.getElementById('enrollLog');
  const t = new Date().toLocaleTimeString();
  el.innerHTML = `<div class="log-entry"><span class="log-time">${escapeHtml(t)}</span>${escapeHtml(msg)}</div>` + el.innerHTML;
}

// Profiles
async function saveProfile() {
  const name = document.getElementById('profileName').value.trim() || 'default';
  const fd = new FormData(); fd.append('name', name);
  const r = await fetch(API + '/api/enroll/save', { method: 'POST', body: fd });
  const d = await r.json();
  r.ok ? toast('success', t('profileSaved', { name })) : toast('error', d.error);
  loadProfileList();
}

async function loadProfile() {
  const name = document.getElementById('profileName').value.trim() || 'default';
  const fd = new FormData(); fd.append('name', name);
  const r = await fetch(API + '/api/enroll/load', { method: 'POST', body: fd });
  const d = await r.json();
  if (r.ok) { toast('success', t('profileLoaded', { name, count: d.keywords })); refreshEnrolled(); }
  else toast('error', d.error);
}

async function loadProfileList() {
  try {
    const r = await fetch(API + '/api/profiles');
    const d = await r.json();
    const el = document.getElementById('profileList');
    el.textContent = d.profiles.length ? t('savedProfiles', { profiles: d.profiles.join(', ') }) : t('noProfiles');
  } catch {}
}

// Detect single
async function detectSingle() {
  let file = audioBlobs.detect;
  if (!file) { toast('error', t('uploadOrRecordFirst')); return; }
  const out = document.getElementById('detectResult');
  setBusy('detectBtn', true, t('detecting'));
  out.innerHTML = '<div class="flex-center" style="padding:40px"><div class="spinner"></div></div>';
  const fd = new FormData();
  fd.append('audio', file, 'audio.wav');
  fd.append('threshold', document.getElementById('detectThr').value);
  fd.append('use_per_class', checkboxFormValue('perClassChk'));
  fd.append('use_close_word_guard', checkboxFormValue('closeWordGuardChk'));
  try {
    const r = await fetch(API + '/api/detect/single', { method: 'POST', body: fd });
    const d = await r.json();
    if (!r.ok) { toast('error', d.error); out.innerHTML = ''; return; }
    const cls = d.detected ? 'detected' : 'rejected';
    const top2 = d.second_label ? `${escapeHtml(d.keyword)} vs ${escapeHtml(d.second_label)}` : escapeHtml(d.keyword);
    out.innerHTML = `<div class="result-card ${cls}">
      <div class="result-keyword">${d.detected ? escapeHtml(d.keyword).toUpperCase() : escapeHtml(t('unknown'))}</div>
      <div class="result-meta">
        <span><span class="badge ${d.detected ? 'badge-success' : 'badge-danger'}">${d.detected ? escapeHtml(t('detected')) : escapeHtml(t('rejected'))}</span></span>
        <span class="text-sm">${escapeHtml(t('dist'))}: <strong>${metricText(d.distance, 4)}</strong></span>
        <span class="text-sm">${escapeHtml(t('thr'))}: <strong>${metricText(d.threshold, 3)}</strong></span>
      </div>
      <div class="result-metrics">
        <div class="result-metric"><strong>${metricText(d.confidence)}</strong><span>${escapeHtml(t('confidence'))}</span></div>
        <div class="result-metric"><strong>${metricText(d.margin, 4)}</strong><span>${escapeHtml(t('top2Margin'))}</span></div>
        <div class="result-metric"><strong>${top2}</strong><span>${escapeHtml(t('topCandidates'))}</span></div>
      </div>
    </div>`;
    pushDetectionHistory({
      detected: d.detected,
      keyword: d.detected ? d.keyword : 'unknown',
      confidence: d.confidence,
      threshold: d.threshold,
      margin: d.margin,
      distance: d.distance,
      state: d.detected ? t('detected') : t('rejected'),
    });
    renderDistBars(d.all_distances, d.threshold);
    if (d.mfcc) renderMFCC(d.mfcc);
  } catch {
    toast('error', t('networkError'));
    out.innerHTML = '';
  } finally {
    setBusy('detectBtn', false);
  }
}

function renderDistBars(dists, thr) {
  const card = document.getElementById('detectChartCard');
  const c = document.getElementById('distBars');
  card.classList.remove('hidden');
  const entries = Object.entries(dists);
  const mx = Math.max(...entries.map(e => e[1]), thr) * 1.2;
  c.innerHTML = entries.map(([w, d]) => {
    const pct = Math.min((d / mx) * 100, 100);
    const cls = d <= thr ? 'match' : 'no-match';
    return `<div class="dist-row">
      <div class="dist-label">${escapeHtml(w)}</div>
      <div class="dist-track"><div class="dist-fill ${cls}" style="width:0%" data-w="${pct}%"></div></div>
      <div class="dist-val">${d.toFixed(3)}</div>
    </div>`;
  }).join('');
  // Animate
  requestAnimationFrame(() => {
    c.querySelectorAll('.dist-fill').forEach(f => { f.style.width = f.dataset.w; });
  });
}

function renderMFCC(mfcc) {
  document.getElementById('mfccCard').classList.remove('hidden');
  const canvas = document.getElementById('mfccCanvas');
  const ctx = canvas.getContext('2d');
  const rows = mfcc.length, cols = mfcc[0].length;
  canvas.width = canvas.offsetWidth * 2;
  canvas.height = canvas.offsetHeight * 2;
  const w = canvas.width, h = canvas.height;
  const cw = w / rows, ch = h / cols;
  let mn = Infinity, mx = -Infinity;
  for (const row of mfcc) for (const v of row) { mn = Math.min(mn, v); mx = Math.max(mx, v); }
  const rng = mx - mn || 1;
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
    const v = (mfcc[i][j] - mn) / rng;
    ctx.fillStyle = `hsl(${(1 - v) * 120}, 80%, ${20 + v * 40}%)`;
    ctx.fillRect(i * cw, (cols - 1 - j) * ch, cw + 1, ch + 1);
  }
}

function pushDetectionHistory(item) {
  detectionHistory.unshift({
    time: new Date(),
    ...item,
  });
  detectionHistory = detectionHistory.slice(0, 12);
  renderDetectionHistory();
}

function clearDetectionHistory() {
  detectionHistory = [];
  renderDetectionHistory();
}

function renderDetectionHistory() {
  const el = document.getElementById('detectHistory');
  if (!el) return;
  if (!detectionHistory.length) {
    el.innerHTML = `<p class="text-sm text-muted">${escapeHtml(t('noDetectionsYet'))}</p>`;
    return;
  }
  el.innerHTML = detectionHistory.map(item => {
    const status = item.detected ? 'badge-success' : 'badge-danger';
    const label = item.detected ? String(item.keyword).toUpperCase() : t('rejected').toUpperCase();
    return `<div class="history-entry">
      <div class="history-main">
        <span class="badge ${status}">${escapeHtml(label)}</span>
        <strong>${escapeHtml(item.state || (item.detected ? t('detected') : t('rejected')))}</strong>
      </div>
      <div class="history-meta">
        <span>${escapeHtml(item.time.toLocaleTimeString())}</span>
        <span>conf ${metricText(item.confidence)}</span>
        <span>${escapeHtml(t('thr'))} ${metricText(item.threshold)}</span>
        <span>margin ${metricText(item.margin, 4)}</span>
      </div>
    </div>`;
  }).join('');
}

// Detect long
let groundTruthFile = null;
let groundTruthLabels = [];
let groundTruthTimingFile = null;
let groundTruthTimings = [];

function parseGroundTruthLabels(text) {
  return String(text || '')
    .split(/\r?\n/)
    .map(line => line.trim())
    .filter(line => line && !line.startsWith('#'))
    .flatMap(line => line.split(','))
    .map(label => label.trim().toLowerCase())
    .filter(Boolean);
}

function parseGroundTruthTimings(text) {
  const raw = JSON.parse(String(text || '{}'));
  const rows = Array.isArray(raw) ? raw : Array.isArray(raw.words) ? raw.words : [];
  const sampleRate = Number(raw.sample_rate || raw.sampleRate || 16000) || 16000;
  return rows.map((item, index) => {
    const label = String(item.label || item.word || item.keyword || '').trim().toLowerCase();
    const startSec = Number.isFinite(Number(item.start_sec))
      ? Number(item.start_sec)
      : Number.isFinite(Number(item.start_ms))
        ? Number(item.start_ms) / 1000
        : Number.isFinite(Number(item.start_sample))
          ? Number(item.start_sample) / sampleRate
          : null;
    const endSec = Number.isFinite(Number(item.end_sec))
      ? Number(item.end_sec)
      : Number.isFinite(Number(item.end_ms))
        ? Number(item.end_ms) / 1000
        : Number.isFinite(Number(item.end_sample))
          ? Number(item.end_sample) / sampleRate
          : null;
    return { label, start_sec: startSec, end_sec: endSec, index };
  })
    .filter(item => item.label && Number.isFinite(item.start_sec) && Number.isFinite(item.end_sec) && item.end_sec > item.start_sec)
    .sort((a, b) => a.start_sec - b.start_sec);
}

function expectedLabelsForDisplay() {
  return groundTruthTimings.length ? groundTruthTimings.map(item => item.label) : groundTruthLabels;
}

function renderGroundTruthPreview() {
  const preview = document.getElementById('gtPreview');
  if (!preview) return;
  if (!groundTruthLabels.length && !groundTruthTimings.length) {
    preview.classList.add('hidden');
    preview.innerHTML = '';
    return;
  }
  const maxVisible = 80;
  const visible = groundTruthLabels.slice(0, maxVisible);
  const remaining = groundTruthLabels.length - visible.length;
  const visibleTimings = groundTruthTimings.slice(0, maxVisible);
  const remainingTimings = groundTruthTimings.length - visibleTimings.length;
  preview.classList.remove('hidden');
  preview.innerHTML = `
    ${groundTruthLabels.length ? `
    <div class="gt-preview-header">
      <div>
        <div class="gt-preview-title">${escapeHtml(t('labelsInFile'))}</div>
        <div class="gt-preview-meta">${escapeHtml(groundTruthFile?.name || 'label.txt')}</div>
      </div>
      <span class="badge badge-neutral">${escapeHtml(t('labelCount', { count: groundTruthLabels.length }))}</span>
    </div>
    <div class="gt-label-list">
      ${visible.map((label, index) => `<span class="gt-label-chip">
        <span class="gt-label-index">${index + 1}</span>
        <span class="gt-label-word">${escapeHtml(label)}</span>
      </span>`).join('')}
      ${remaining > 0 ? `<span class="gt-label-chip"><span class="gt-label-word">${escapeHtml(t('moreLabels', { count: remaining }))}</span></span>` : ''}
    </div>` : ''}
    ${groundTruthTimings.length ? `
    <div class="gt-preview-header ${groundTruthLabels.length ? 'mt-12' : ''}">
      <div>
        <div class="gt-preview-title">${escapeHtml(t('timingsInFile'))}</div>
        <div class="gt-preview-meta">${escapeHtml(groundTruthTimingFile?.name || 'timings.json')}</div>
      </div>
      <span class="badge badge-neutral">${escapeHtml(t('timingCount', { count: groundTruthTimings.length }))}</span>
    </div>
    <div class="gt-label-list">
      ${visibleTimings.map((item, index) => `<span class="gt-label-chip gt-timing-chip">
        <span class="gt-label-index">${index + 1}</span>
        <span class="gt-label-word">${escapeHtml(item.label)}</span>
        <span class="gt-time-range">${escapeHtml(formatSeconds(item.start_sec))}-${escapeHtml(formatSeconds(item.end_sec))}</span>
      </span>`).join('')}
      ${remainingTimings > 0 ? `<span class="gt-label-chip"><span class="gt-label-word">${escapeHtml(t('moreLabels', { count: remainingTimings }))}</span></span>` : ''}
    </div>` : ''}`;
}

async function handleGroundTruthFile(file) {
  if (!file) return;
  const gtDrop = document.getElementById('gtDrop');
  groundTruthFile = file;
  const text = await file.text();
  groundTruthLabels = parseGroundTruthLabels(text);
  if (gtDrop) gtDrop.querySelector('p').textContent = t('loadedFile', { name: file.name });
  renderGroundTruthPreview();
  if (groundTruthLabels.length) {
    toast('success', t('labelsLoaded', { count: groundTruthLabels.length, name: file.name }));
  } else {
    toast('error', t('invalidGroundTruth'));
  }
}

async function handleTimingFile(file) {
  if (!file) return;
  const timingDrop = document.getElementById('timingDrop');
  groundTruthTimingFile = file;
  const text = await file.text();
  groundTruthTimings = parseGroundTruthTimings(text);
  if (timingDrop) timingDrop.querySelector('p').textContent = t('loadedFile', { name: file.name });
  renderGroundTruthPreview();
  if (groundTruthTimings.length) {
    toast('success', t('timingsLoaded', { count: groundTruthTimings.length, name: file.name }));
  } else {
    toast('error', t('invalidTimingJson'));
  }
}

function renderSequenceBadges(labels, emptyText, badgeClass = 'badge-neutral') {
  if (!labels || !labels.length) {
    return `<span class="text-muted">${escapeHtml(emptyText)}</span>`;
  }
  return labels.map(w => `<span class="badge ${badgeClass}">${escapeHtml(w)}</span>`).join(' ');
}

function renderNumberedExpectedLabels(labels, offset = 0) {
  return labels.map((label, index) => `<span class="gt-label-chip">
    <span class="gt-label-index">${offset + index + 1}</span>
    <span class="gt-label-word">${escapeHtml(label)}</span>
  </span>`).join('');
}

function formatSeconds(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return '-';
  return `${n.toFixed(2)}s`;
}

function intervalOverlap(a0, a1, b0, b1) {
  return Math.max(0, Math.min(Number(a1), Number(b1)) - Math.max(Number(a0), Number(b0)));
}

function timingMatchForDetection(segment) {
  if (!groundTruthTimings.length) return null;
  let best = null;
  let bestOverlap = 0;
  for (const item of groundTruthTimings) {
    const overlap = intervalOverlap(segment.t0, segment.t1, item.start_sec, item.end_sec);
    if (overlap > bestOverlap) {
      best = item;
      bestOverlap = overlap;
    }
  }
  if (!best || bestOverlap <= 0) return null;
  return { ...best, overlap_sec: bestOverlap };
}

function timelineMinWidth(duration, expectedCount, detectedCount) {
  const byDuration = Math.ceil(Number(duration || 0) * 52);
  const byCount = Math.max(Number(expectedCount || 0), Number(detectedCount || 0)) * 76;
  return Math.max(920, byDuration, byCount);
}

function timelineSegmentLabel(label, widthPct) {
  if (widthPct < 1.8) return escapeHtml(String(label || '').slice(0, 1));
  if (widthPct < 4.2) return escapeHtml(String(label || '').slice(0, 3));
  return escapeHtml(label || '');
}

function renderExpectedTimeline(timings, duration, minWidthPx) {
  if (!timings.length || !duration) return '';
  return `<div class="timeline-caption">${escapeHtml(t('expectedTimeline'))}</div>
    <div class="timeline-scroll" tabindex="0">
    <div class="timeline-track timeline-track-expected" style="min-width:${minWidthPx}px">
      ${timings.map(item => {
        const left = Math.max(0, Math.min(100, (item.start_sec / duration) * 100));
        const width = Math.max(((item.end_sec - item.start_sec) / duration) * 100, 1.5);
        return `<div class="timeline-seg expected" style="left:${left}%;width:${width}%"
          title="${escapeHtml(formatSeconds(item.start_sec))}-${escapeHtml(formatSeconds(item.end_sec))}: ${escapeHtml(item.label)}">${timelineSegmentLabel(item.label, width)}</div>`;
      }).join('')}
    </div></div>
    <div class="timeline-caption mt-8">${escapeHtml(t('detectedTimeline'))}</div>`;
}

function renderDetectedTimeline(results, duration, minWidthPx) {
  if (!results.length || !duration) return '';
  return `<div class="timeline-scroll" tabindex="0">
    <div class="timeline-track" style="min-width:${minWidthPx}px">
      ${results.map(s => {
        const left = Math.max(0, Math.min(100, (s.t0 / duration) * 100));
        const width = Math.max(((s.t1 - s.t0) / duration) * 100, 1.5);
        const label = s.detected ? s.keyword : '?';
        return `<div class="timeline-seg ${s.detected ? 'det' : 'unk'}" style="left:${left}%;width:${width}%"
          title="${escapeHtml(s.t0)}s-${escapeHtml(s.t1)}s: ${escapeHtml(label)} (d=${escapeHtml(s.distance)})">${timelineSegmentLabel(label, width)}</div>`;
      }).join('')}
    </div>
  </div>`;
}

function pctText(numerator, denominator) {
  if (!denominator) return '-';
  return `${(numerator / denominator * 100).toFixed(1)}%`;
}

function uniqueLabels(labels) {
  return [...new Set((labels || []).map(label => String(label || '').toLowerCase()).filter(Boolean))];
}

function renderResultMetric(label, value, tone = '') {
  return `<div class="result-summary-card ${tone}">
    <div class="result-summary-value">${escapeHtml(value)}</div>
    <div class="result-summary-label">${escapeHtml(label)}</div>
  </div>`;
}

function renderTopCandidates(candidates) {
  if (!candidates || !candidates.length) return '<span class="text-muted">-</span>';
  return candidates.map((c, idx) =>
    `<span class="candidate-line ${idx === 0 ? 'primary' : ''}">${idx + 1}. ${escapeHtml(c.word)}
      <span class="font-mono">(${escapeHtml(c.dist)})</span></span>`
  ).join('');
}

function predictedWord(segment) {
  return segment?.detected ? segment.keyword : 'unknown';
}

function findMissedExpectedTimings(results) {
  if (!groundTruthTimings.length) return [];
  return groundTruthTimings.map(item => {
    const overlaps = (results || [])
      .map((segment, index) => ({
        index,
        segment,
        overlap_sec: intervalOverlap(segment.t0, segment.t1, item.start_sec, item.end_sec),
      }))
      .filter(row => row.overlap_sec > 0)
      .sort((a, b) => b.overlap_sec - a.overlap_sec);
    const correct = overlaps.find(row => predictedWord(row.segment) === item.label);
    if (correct) return null;
    const best = overlaps[0] || null;
    let reason = t('missReasonNoOverlap');
    if (best) {
      const predicted = predictedWord(best.segment);
      if (best.segment.detected) {
        reason = t('missReasonWrongPrediction', { predicted });
      } else {
        const distance = Number(best.segment.distance);
        const threshold = Number(best.segment.threshold);
        const margin = Number(best.segment.margin);
        const acceptMargin = Number(best.segment.accept_margin);
        if (Number.isFinite(distance) && Number.isFinite(threshold) && distance > threshold) {
          reason = t('missReasonRejected');
        } else if (Number.isFinite(acceptMargin) && acceptMargin > 0 && Number.isFinite(margin) && margin < acceptMargin) {
          reason = t('missReasonLowMargin', { predicted: best.segment.best_label || best.segment.top_3?.[0]?.word || predicted, margin: metricText(margin, 4) });
        } else {
          reason = t('missReasonRejectedGeneric');
        }
      }
    }
    return { ...item, reason, overlap: best };
  }).filter(Boolean);
}

function renderMissedExpectedCard(item) {
  const overlap = item.overlap;
  const segment = overlap?.segment || null;
  const predicted = segment ? predictedWord(segment) : '-';
  const overlapInfo = segment
    ? `<div class="detection-detail-grid miss-detail-grid">
      <div><span class="mini-label">${escapeHtml(t('overlappedDetection'))}</span><strong>#${Number(overlap.index) + 1}</strong></div>
      <div><span class="mini-label">${escapeHtml(t('predicted'))}</span><strong>${escapeHtml(predicted)}</strong></div>
      <div><span class="mini-label">${escapeHtml(t('l2Dist'))}</span><strong class="font-mono">${escapeHtml(segment.distance)}</strong></div>
      <div><span class="mini-label">${escapeHtml(t('threshold'))}</span><strong class="font-mono">${escapeHtml(segment.threshold)}</strong></div>
      <div><span class="mini-label">${escapeHtml(t('margin'))}</span><strong class="font-mono">${escapeHtml(metricText(segment.margin, 4))}</strong></div>
      <div><span class="mini-label">${escapeHtml(t('acceptMargin'))}</span><strong class="font-mono">${escapeHtml(metricText(segment.accept_margin, 4))}</strong></div>
    </div>
    ${segment.top_3?.length ? `<div class="candidate-box">
      <div class="mini-label">${escapeHtml(t('top3Candidates'))}</div>
      ${renderTopCandidates(segment.top_3)}
    </div>` : ''}`
    : '';
  return `<article class="detection-card match-miss">
    <div class="detection-card-head">
      <div>
        <div class="detection-index">#E${Number(item.index) + 1}</div>
        <div class="detection-time">${formatSeconds(item.start_sec)} - ${formatSeconds(item.end_sec)}</div>
      </div>
      <span class="badge badge-danger">${escapeHtml(t('missedExpected'))}</span>
    </div>
    <div class="detection-main">
      <div>
        <div class="mini-label">${escapeHtml(t('predicted'))}</div>
        <div class="detection-word predicted">${escapeHtml(predicted)}</div>
      </div>
      <div>
        <div class="mini-label">${escapeHtml(t('expected'))}</div>
        <div class="detection-word expected">${escapeHtml(item.label)}</div>
      </div>
    </div>
    <div class="candidate-box candidate-box-warning">
      <div class="mini-label">${escapeHtml(t('missReason'))}</div>
      ${escapeHtml(item.reason || t('noDetectionForExpected'))}
    </div>
    ${overlapInfo}
  </article>`;
}

function renderDetectionCard(s, i, timingMatches, hasGT, timingMode, gtLabels) {
  const predicted = s.detected ? s.keyword : 'unknown';
  const expectedTiming = timingMode ? timingMatches[i] : null;
  const expected = timingMode
    ? expectedTiming?.label || null
    : hasGT && i < gtLabels.length ? gtLabels[i] : null;
  const match = expected !== null ? predicted === expected : null;
  const statusClass = match === true ? 'match-ok' : match === false ? 'match-err' : hasGT ? 'match-warn' : 'match-neutral';
  const matchBadge = match === true ? '<span class="badge badge-success">OK</span>'
    : match === false ? '<span class="badge badge-danger">ERR</span>'
      : hasGT ? `<span class="badge badge-warn">${escapeHtml(t('noExpectedOverlap'))}</span>` : '';
  const expectedText = expected
    ? timingMode && expectedTiming ? `${expected} (${formatSeconds(expectedTiming.start_sec)}-${formatSeconds(expectedTiming.end_sec)})` : expected
    : '-';
  return `<article class="detection-card ${statusClass}">
    <div class="detection-card-head">
      <div>
        <div class="detection-index">#${i + 1}</div>
        <div class="detection-time">${escapeHtml(s.t0)}s - ${escapeHtml(s.t1)}s</div>
      </div>
      ${matchBadge}
    </div>
    <div class="detection-main">
      <div>
        <div class="mini-label">${escapeHtml(t('predicted'))}</div>
        <div class="detection-word predicted">${escapeHtml(predicted)}</div>
      </div>
      ${hasGT ? `<div>
        <div class="mini-label">${escapeHtml(t('expected'))}</div>
        <div class="detection-word expected">${escapeHtml(expectedText)}</div>
      </div>` : ''}
    </div>
    <div class="detection-detail-grid">
      <div><span class="mini-label">${escapeHtml(t('l2Dist'))}</span><strong class="font-mono">${escapeHtml(s.distance)}</strong></div>
      <div><span class="mini-label">${escapeHtml(t('threshold'))}</span><strong class="font-mono">${escapeHtml(s.threshold)}</strong></div>
      <div><span class="mini-label">${escapeHtml(t('acceptMargin'))}</span><strong class="font-mono">${escapeHtml(metricText(s.accept_margin, 4))}</strong></div>
      <div><span class="mini-label">${escapeHtml(t('status'))}</span><strong>${s.detected ? escapeHtml(t('detected')) : escapeHtml(t('rejected'))}</strong></div>
    </div>
    <div class="candidate-box">
      <div class="mini-label">${escapeHtml(t('top3Candidates'))}</div>
      ${renderTopCandidates(s.top_3 || [])}
    </div>
  </article>`;
}

function renderDetectionCards(results, timingMatches, hasGT, timingMode, gtLabels) {
  if (!results.length && !timingMode) return `<div class="empty-state compact"><p>${escapeHtml(t('noKeywordsDetected'))}</p></div>`;
  if (timingMode) {
    const cards = [];
    results.forEach((s, i) => {
      cards.push({
        time: Number(s.t0) || 0,
        html: renderDetectionCard(s, i, timingMatches, hasGT, timingMode, gtLabels),
      });
    });
    findMissedExpectedTimings(results).forEach(item => {
      cards.push({
        time: Number(item.start_sec) || 0,
        html: renderMissedExpectedCard(item),
      });
    });
    if (!cards.length) return `<div class="empty-state compact"><p>${escapeHtml(t('noKeywordsDetected'))}</p></div>`;
    cards.sort((a, b) => a.time - b.time);
    return `<div class="detection-card-grid">${cards.map(card => card.html).join('')}</div>`;
  }
  return `<div class="detection-card-grid">
    ${results.map((s, i) => renderDetectionCard(s, i, timingMatches, hasGT, timingMode, gtLabels)).join('')}
  </div>`;
}

async function detectLong() {
  const fileInput = document.getElementById('longFile');
  let file = fileInput.files.length ? fileInput.files[0] : audioBlobs.long;
  if (!file) { toast('error', t('uploadAudioFirst')); return; }
  const out = document.getElementById('longResult');
  const accDiv = document.getElementById('longAccuracy');
  setBusy('longDetectBtn', true, t('analyzing'));
  out.innerHTML = '<div class="flex-center" style="padding:40px"><div class="spinner"></div></div>';
  accDiv.classList.add('hidden');

  const fd = new FormData();
  fd.append('audio', file, 'audio.wav');
  fd.append('threshold', document.getElementById('longThr').value);
  fd.append('use_per_class', checkboxFormValue('longPerClass'));
  fd.append('use_close_word_guard', checkboxFormValue('longCloseWordGuard'));
  fd.append('seg_method', document.getElementById('longSeg').value);
  fd.append('min_duration_ms', document.getElementById('longMinDur').value);
  try {
    const r = await fetch(API + '/api/detect/long', { method: 'POST', body: fd });
    const d = await r.json();
    if (!r.ok) { toast('error', d.error); out.innerHTML = ''; return; }

    const timingMode = groundTruthTimings.length > 0;
    const displayLabels = expectedLabelsForDisplay();
    const gtLabels = displayLabels.length ? displayLabels : null;
    const timingMatches = timingMode ? d.results.map(s => timingMatchForDetection(s)) : [];
    const gtComparable = !timingMode && gtLabels && gtLabels.length === d.results.length;
    const hasGT = Boolean(gtLabels && gtLabels.length > 0);
    const expectedLabels = timingMode
      ? groundTruthTimings.map(item => item.label)
      : hasGT ? gtLabels : [];
    const expectedTotal = expectedLabels.length;
    const enrolledExpectedLabels = enrolledKeywordsState.size
      ? expectedLabels.filter(label => enrolledKeywordsState.has(String(label).toLowerCase()))
      : [];
    const outOfEnrollmentExpected = enrolledKeywordsState.size
      ? expectedLabels.filter(label => !enrolledKeywordsState.has(String(label).toLowerCase()))
      : [];
    const outOfEnrollmentLabels = enrolledKeywordsState.size
      ? uniqueLabels(outOfEnrollmentExpected)
      : [];
    const displayedOutLabels = outOfEnrollmentLabels.length > 12
      ? `${outOfEnrollmentLabels.slice(0, 12).join(', ')}...`
      : outOfEnrollmentLabels.join(', ');

    let accCorrect = 0;
    let enrolledCorrect = 0;
    let enrolledTotal = enrolledExpectedLabels.length;
    if (timingMode) {
      const correctExpected = new Set();
      const correctEnrolledExpected = new Set();
      d.results.forEach((s, i) => {
        const expected = timingMatches[i];
        const predicted = s.detected ? s.keyword : 'unknown';
        if (expected && predicted === expected.label) {
          correctExpected.add(expected.index);
          if (!enrolledKeywordsState.size || enrolledKeywordsState.has(String(expected.label).toLowerCase())) {
            correctEnrolledExpected.add(expected.index);
          }
        }
      });
      accCorrect = correctExpected.size;
      enrolledCorrect = correctEnrolledExpected.size;
      if (!enrolledKeywordsState.size) enrolledTotal = groundTruthTimings.length;
    } else if (gtComparable) {
      for (let i = 0; i < d.results.length; i++) {
        const predicted = d.results[i].detected ? d.results[i].keyword : 'unknown';
        const expected = gtLabels[i];
        if (predicted === expected) {
          accCorrect++;
          if (!enrolledKeywordsState.size || enrolledKeywordsState.has(String(expected).toLowerCase())) {
            enrolledCorrect++;
          }
        }
      }
      if (!enrolledKeywordsState.size) enrolledTotal = expectedTotal;
    }

    const resultUnit = d.engine === 'robust_state_machine' ? t('detections').toLowerCase() : t('segments').toLowerCase();
    const closeWordGuardOn = d.settings?.close_word_guard !== false;
    const perClassOn = d.settings?.use_per_class !== false;
    const acceptMargin = metricText(d.settings?.accept_margin, 4);
    const accuracyComparable = timingMode || gtComparable || !hasGT;
    const allAccuracy = accuracyComparable ? pctText(accCorrect, expectedTotal) : '-';
    const enrolledAccuracy = accuracyComparable ? pctText(enrolledCorrect, enrolledTotal) : '-';
    const accuracyTone = accuracyComparable && expectedTotal && accCorrect / expectedTotal >= 0.8 ? 'good' : accuracyComparable && expectedTotal ? 'warn' : '';
    const enrolledTone = accuracyComparable && enrolledTotal && enrolledCorrect / enrolledTotal >= 0.8 ? 'good' : accuracyComparable && enrolledTotal ? 'warn' : '';
    const minWidthPx = timelineMinWidth(d.duration, groundTruthTimings.length, d.results.length);
    let html = `<div class="card">
      <div class="card-header"><div class="card-title">${escapeHtml(t('results'))}</div>
        <span class="badge badge-neutral">${d.duration}s - ${d.segments} ${resultUnit}</span></div>
      <div class="result-summary-grid mb-16">
        ${renderResultMetric(t('duration'), `${d.duration}s`)}
        ${renderResultMetric(t('expectedCount'), String(expectedTotal || '-'))}
        ${renderResultMetric(t('detectionCount'), String(d.results.length))}
        ${renderResultMetric(t('matched'), accuracyComparable && expectedTotal ? `${accCorrect}/${expectedTotal}` : '-')}
        ${renderResultMetric(t('allAccuracy'), allAccuracy, accuracyTone)}
        ${renderResultMetric(t('enrolledOnlyAccuracy'), enrolledAccuracy, enrolledTone)}
        ${renderResultMetric(t('perClassThreshold'), perClassOn ? t('perClassOn') : t('perClassOff'), perClassOn ? '' : 'warn')}
        ${renderResultMetric(t('closeWordGuard'), closeWordGuardOn ? t('closeWordGuardOn') : t('closeWordGuardOff'), closeWordGuardOn ? '' : 'warn')}
        ${renderResultMetric(t('acceptMargin'), acceptMargin, acceptMargin === '0.0000' ? 'warn' : '')}
        ${renderResultMetric(t('outOfEnrollment'), outOfEnrollmentExpected.length ? String(outOfEnrollmentExpected.length) : '0', outOfEnrollmentExpected.length ? 'warn' : 'good')}
      </div>
      ${outOfEnrollmentLabels.length ? `<div class="alert alert-warning mb-16">
        ${escapeHtml(t('outOfEnrollmentWarning', { count: outOfEnrollmentExpected.length, labels: displayedOutLabels }))}
      </div>` : hasGT && enrolledKeywordsState.size ? `<div class="alert mb-16">${escapeHtml(t('noOutOfEnrollment'))}</div>` : ''}
      <div class="sequence-panel mb-16">
        <div class="sequence-row">
          <span class="sequence-label">${escapeHtml(t('predictedSequence'))}</span>
          <span class="sequence-items">${renderSequenceBadges(d.sequence, t('noKeywordsDetected'), 'badge-success')}</span>
        </div>
        ${gtLabels && gtLabels.length ? `<div class="sequence-row">
          <span class="sequence-label">${escapeHtml(t('expectedSequence'))}</span>
          <span class="sequence-items">${renderSequenceBadges(gtLabels, '-', 'badge-neutral')}</span>
        </div>` : ''}
      </div>`;
    if (!timingMode && gtLabels && gtLabels.length > 0 && !gtComparable) {
      html += `<div class="alert alert-warning mb-16">
        ${escapeHtml(t('expectedMismatch', { expected: gtLabels.length, unit: resultUnit, actual: d.results.length }))}
      </div>`;
    }
    if (timingMode) {
      html += `<div class="alert mb-16">${escapeHtml(t('timingMatchMode'))}</div>`;
    }
    if (d.results.length && d.duration > 0) {
      if (timingMode) {
        html += renderExpectedTimeline(groundTruthTimings, d.duration, minWidthPx);
      }
      html += renderDetectedTimeline(d.results, d.duration, minWidthPx);
    }

    html += renderDetectionCards(d.results, timingMatches, hasGT, timingMode, gtLabels || []);
    if (!timingMode && hasGT && gtLabels.length > d.results.length) {
      const remainingLabels = gtLabels.slice(d.results.length);
      html += `<div class="comparison-note mt-16">
        <div class="comparison-note-title">${escapeHtml(t('expectedNotCompared'))}</div>
        <div class="gt-label-list">${renderNumberedExpectedLabels(remainingLabels, d.results.length)}</div>
      </div>`;
    } else if (!timingMode && hasGT && d.results.length > gtLabels.length) {
      const extras = d.results.slice(gtLabels.length).map(s => s.detected ? s.keyword : 'unknown');
      html += `<div class="comparison-note mt-16">
        <div class="comparison-note-title">${escapeHtml(t('extraDetections'))}</div>
        <div class="gt-label-list">${renderNumberedExpectedLabels(extras, gtLabels.length)}</div>
      </div>`;
    }
    html += '</div>';
    out.innerHTML = html;
  } catch {
    toast('error', t('networkError'));
  } finally {
    setBusy('longDetectBtn', false);
  }
}

// Streaming
let streamWs = null;
let streamAudioCtx = null;
let streamSource = null;
let streamProcessor = null;
let streamAnalyser = null;
let streamAnimFrame = null;
let streamStartTime = null;
let streamTimerInterval = null;
let streamDetectionCount = 0;
let isStreaming = false;

async function toggleStreaming() {
  if (isStreaming) { stopStreaming(); return; }

  const btn = document.getElementById('streamBtn');
  const status = document.getElementById('streamStatus');

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true } });

    streamAudioCtx = new AudioContext({ sampleRate: 16000 });
    streamSource = streamAudioCtx.createMediaStreamSource(stream);
    streamAnalyser = streamAudioCtx.createAnalyser();
    streamAnalyser.fftSize = 2048;
    streamSource.connect(streamAnalyser);

    // ScriptProcessor to capture PCM chunks
    streamProcessor = streamAudioCtx.createScriptProcessor(4096, 1, 1);
    streamSource.connect(streamProcessor);
    streamProcessor.connect(streamAudioCtx.destination);

    // WebSocket
    const wsUrl = `ws://${location.host}/ws/stream`;
    streamWs = new WebSocket(wsUrl);
    streamWs.binaryType = 'arraybuffer';

    streamWs.onopen = () => {
      isStreaming = true;
      btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="20" height="20"><rect x="6" y="6" width="12" height="12" rx="2"/></svg> ${escapeHtml(t('stopStreaming'))}`;
      btn.classList.add('recording');
      status.textContent = t('listening');
      status.className = 'badge badge-success';
      streamDetectionCount = 0;
      streamStartTime = Date.now();
      document.getElementById('statDetections').textContent = '0';
      document.getElementById('statLastWord').textContent = '-';
      document.getElementById('streamFeed').innerHTML = '';
      startStreamTimer();
      drawStreamWaveform();
      toast('success', t('streamingStarted') || t('startStreaming'));
    };

    streamWs.onmessage = (ev) => {
      const d = JSON.parse(ev.data);
      if (d.detected) {
        streamDetectionCount++;
        document.getElementById('statDetections').textContent = streamDetectionCount;
        document.getElementById('statLastWord').textContent = d.keyword.toUpperCase();
      }

      // Add to feed
      const feed = document.getElementById('streamFeed');
      const timeText = new Date().toLocaleTimeString();
      const top3 = (d.top_3 || []).map(c => `${escapeHtml(c.word)}(${escapeHtml(c.dist)})`).join(', ');
      const cls = d.detected ? 'badge-success' : 'badge-neutral';
      const state = d.state || (d.detected ? t('detected') : t('listening'));
      const entry = document.createElement('div');
      entry.className = 'log-entry';
      entry.innerHTML = `<span class="log-time">${escapeHtml(timeText)}</span>
        <span class="badge ${cls}" style="margin-right:6px">${d.detected ? escapeHtml(d.keyword).toUpperCase() : escapeHtml(t('listening'))}</span>
        <span class="text-xs text-muted">${escapeHtml(state)} | L2=${escapeHtml(d.distance)} ${escapeHtml(t('thr'))}=${escapeHtml(d.threshold)} conf=${metricText(d.confidence)} margin=${metricText(d.margin, 4)}</span>
        <span class="text-xs text-muted" style="margin-left:6px">[${top3}]</span>`;
      feed.insertBefore(entry, feed.firstChild);

      // Flash effect on detection
      if (d.detected) {
        pushDetectionHistory(d);
        entry.style.background = 'rgba(15,118,110,.12)';
        entry.style.borderLeft = '3px solid var(--accent-500)';
        entry.style.padding = '8px 12px';
        entry.style.borderRadius = 'var(--radius-sm)';
        entry.style.marginBottom = '6px';
      }
    };

    streamWs.onerror = () => toast('error', t('webSocketError'));
    streamWs.onclose = () => { if (isStreaming) stopStreaming(); };

    // Send audio chunks
    streamProcessor.onaudioprocess = (e) => {
      if (!streamWs || streamWs.readyState !== WebSocket.OPEN) return;
      const data = e.inputBuffer.getChannelData(0);
      streamWs.send(data.buffer.slice(0));
    };

  } catch (err) {
    toast('error', t('micDenied'));
    console.error(err);
  }
}

function stopStreaming() {
  isStreaming = false;
  const btn = document.getElementById('streamBtn');
  const status = document.getElementById('streamStatus');

  if (streamWs) { streamWs.close(); streamWs = null; }
  if (streamProcessor) { streamProcessor.disconnect(); streamProcessor = null; }
  if (streamSource) { streamSource.disconnect(); streamSource = null; }
  if (streamAnalyser) { streamAnalyser = null; }
  if (streamAudioCtx) {
    streamAudioCtx.close();
    streamAudioCtx = null;
  }
  cancelAnimationFrame(streamAnimFrame);
  clearInterval(streamTimerInterval);

  btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="20" height="20"><path d="M2 10v3"/><path d="M6 6v11"/><path d="M10 3v18"/><path d="M14 8v7"/><path d="M18 5v13"/><path d="M22 10v3"/></svg> ${escapeHtml(t('startStreaming'))}`;
  btn.classList.remove('recording');
  status.textContent = t('stopped');
  status.className = 'badge badge-danger';
  toast('info', t('streamingStopped', { count: streamDetectionCount }));
}

function startStreamTimer() {
  streamTimerInterval = setInterval(() => {
    const elapsed = Math.floor((Date.now() - streamStartTime) / 1000);
    const m = Math.floor(elapsed / 60);
    const s = elapsed % 60;
    document.getElementById('statElapsed').textContent = m > 0 ? `${m}m ${s}s` : `${s}s`;
  }, 1000);
}

function drawStreamWaveform() {
  const canvas = document.getElementById('streamWaveform');
  if (!canvas || !streamAnalyser) return;
  const ctx = canvas.getContext('2d');
  canvas.width = canvas.offsetWidth * 2;
  canvas.height = canvas.offsetHeight * 2;
  const w = canvas.width, h = canvas.height;
  const data = new Uint8Array(streamAnalyser.frequencyBinCount);

  function draw() {
    streamAnimFrame = requestAnimationFrame(draw);
    if (!streamAnalyser) return;
    streamAnalyser.getByteTimeDomainData(data);
    ctx.fillStyle = cssVar('--bg-input') || '#f8fafc';
    ctx.fillRect(0, 0, w, h);
    ctx.lineWidth = 2;
    ctx.strokeStyle = cssVar('--accent-500') || '#0f766e';
    ctx.beginPath();
    const sliceW = w / data.length;
    for (let i = 0; i < data.length; i++) {
      const y = (data[i] / 128.0) * h / 2;
      i === 0 ? ctx.moveTo(0, y) : ctx.lineTo(i * sliceW, y);
    }
    ctx.stroke();
  }
  draw();
}

// Model info
function getModelProfile(profileId) {
  return modelProfilesState.profiles.find(profile => profile.id === profileId) || null;
}

function renderModelProfileStatus(profile) {
  if (!profile) return t('noActiveProfile');
  const existsText = profile.exists ? t('readyLower') : t('missingCheckpoint');
  return `${profile.short_label || profile.label} - ${profile.checkpoint_name} - ${existsText}`;
}

function renderProfileMetrics(profile) {
  const metrics = Array.isArray(profile.metrics) ? profile.metrics.slice(0, 3) : [];
  if (!metrics.length) return '';
  return `<div class="model-option-metrics">${metrics.map(metric => `
    <div class="model-mini-metric">
      <div class="model-mini-value">${escapeHtml(metric.value)}</div>
      <div class="model-mini-label">${escapeHtml(metric.label)}</div>
    </div>`).join('')}</div>`;
}

function renderModelCard(profile, compact = false) {
  const active = profile.id === modelProfilesState.active;
  const missing = !profile.exists;
  const classes = ['model-option-card'];
  if (active) classes.push('active');
  if (missing) classes.push('missing');
  const badgeClass = active ? 'badge-success' : (missing ? 'badge-danger' : 'badge-neutral');
  const badge = active ? t('active') : (missing ? t('missing') : t('readyUpper'));
  const description = compact ? profileText(profile, 'notes') : profileText(profile, 'description');
  const threshold = profile.threshold_hint !== null && profile.threshold_hint !== undefined
    ? `${t('thr')} ${Number(profile.threshold_hint).toFixed(2)}`
    : t('thrAuto');
  return `<button type="button" class="${classes.join(' ')}" data-model-profile-id="${escapeHtml(profile.id)}"
      ${active ? 'aria-current="true"' : ''} ${missing ? 'disabled' : ''}>
    <div class="model-option-head">
      <div>
        <div class="model-option-title">${escapeHtml(profile.short_label || profile.label)}</div>
        <div class="model-option-desc">${escapeHtml(description || '')}</div>
      </div>
      <span class="badge ${badgeClass}">${badge}</span>
    </div>
    <div class="model-option-meta">
      <span class="badge badge-neutral">${escapeHtml(profile.checkpoint_name)}</span>
      <span class="badge badge-neutral">${escapeHtml(profile.feature_type || 'auto')}</span>
      <span class="badge badge-neutral">${escapeHtml(threshold)}</span>
    </div>
    ${renderProfileMetrics(profile)}
  </button>`;
}

function setModelProfileError(message) {
  const topError = document.getElementById('modelTopError');
  const infoError = document.getElementById('modelProfileError');
  const retry = `<button class="btn btn-ghost btn-sm mt-8" type="button" onclick="loadModelProfiles()">${escapeHtml(t('reloadProfiles'))}</button>`;
  [topError, infoError].forEach(el => {
    if (!el) return;
    el.classList.remove('hidden');
    el.innerHTML = `${escapeHtml(message)}${retry}`;
  });
}

function clearModelProfileError() {
  ['modelTopError', 'modelProfileError'].forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    el.classList.add('hidden');
    el.innerHTML = '';
  });
}

function wireModelCards() {
  document.querySelectorAll('[data-model-profile-id]').forEach(card => {
    card.addEventListener('click', () => requestModelSwitch(card.dataset.modelProfileId));
  });
}

function renderModelProfiles() {
  const profiles = modelProfilesState.profiles;
  const active = getModelProfile(modelProfilesState.active);
  const quick = document.getElementById('modelQuickCards');
  const detail = document.getElementById('modelProfileCards');
  const select = document.getElementById('modelProfileSelect');
  const status = document.getElementById('modelProfileStatus');
  const warning = document.getElementById('modelProfileWarning');
  const activeTitle = document.getElementById('activeModelTitle');

  const topProfiles = profiles.filter(profile => ['top500_epoch13', 'microset_epoch05'].includes(profile.id));
  if (quick) {
    quick.innerHTML = topProfiles.length
      ? topProfiles.map(profile => renderModelCard(profile, true)).join('')
      : `<div class="model-card-skeleton">${escapeHtml(modelProfilesState.loading ? t('loadingCheckpoints') : t('noDemoCheckpoints'))}</div>`;
  }
  if (detail) {
    detail.innerHTML = profiles.length
      ? profiles.map(profile => renderModelCard(profile, false)).join('')
      : `<div class="model-card-skeleton">${escapeHtml(modelProfilesState.loading ? t('loadingCheckpoints') : t('noModelProfiles'))}</div>`;
  }
  if (select) {
    select.innerHTML = profiles.length ? profiles.map(profile => {
      const disabled = profile.exists ? '' : ' disabled';
      const selected = profile.id === modelProfilesState.active ? ' selected' : '';
      const suffix = profile.exists ? '' : ` (${t('missingCheckpoint')})`;
      return `<option value="${escapeHtml(profile.id)}"${selected}${disabled}>${escapeHtml((profile.short_label || profile.label) + suffix)}</option>`;
    }).join('') : `<option value="">${escapeHtml(t('loadingCheckpoints'))}</option>`;
  }
  if (status) status.textContent = renderModelProfileStatus(active);
  if (warning) warning.classList.toggle('hidden', profiles.length <= 1);
  if (activeTitle) activeTitle.textContent = active ? (active.short_label || active.label) : t('noActiveModel');
  wireModelCards();
}

async function loadModelProfiles() {
  try {
    clearModelProfileError();
    const r = await fetch(API + '/api/model/profiles', { cache: 'no-store' });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || t('couldNotLoadProfiles'));
    modelProfilesState = {
      active: d.active,
      profiles: d.profiles || [],
      canRebuildOnSwitch: Boolean(d.can_rebuild_on_switch),
      loading: false,
    };
    renderModelProfiles();
  } catch (err) {
    modelProfilesState = { ...modelProfilesState, loading: false };
    console.error('loadModelProfiles failed', err);
    setModelProfileError(t('couldNotLoadProfiles'));
    const status = document.getElementById('modelProfileStatus');
    if (status) status.textContent = t('profileApiFailed');
  }
}

function requestModelSwitch(profileId) {
  const profile = getModelProfile(profileId);
  if (!profile) {
    toast('error', t('modelProfileNotFound'));
    return;
  }
  if (!profile.exists) {
    toast('error', t('checkpointMissing'));
    return;
  }
  if (profile.id === modelProfilesState.active) {
    toast('info', t('modelAlreadyActive'));
    return;
  }
  pendingModelProfileId = profileId;
  openModelSwitchModal(profile);
}

function openModelSwitchModal(profile) {
  const modal = document.getElementById('modelSwitchModal');
  const summary = document.getElementById('modelSwitchSummary');
  const rebuildBtn = document.getElementById('modelConfirmRebuildBtn');
  const status = document.getElementById('modelSwitchModalStatus');
  if (!modal || !summary) return;

  summary.innerHTML = `
    <div class="model-option-title">${escapeHtml(profile.label)}</div>
    <p class="model-option-desc mt-4">${escapeHtml(profileText(profile, 'description') || profileText(profile, 'notes') || '')}</p>
    <div class="model-option-meta mt-8">
      <span class="badge badge-neutral">${escapeHtml(profile.checkpoint_name)}</span>
      <span class="badge badge-neutral">${escapeHtml(profile.model_family)}</span>
      <span class="badge badge-neutral">${escapeHtml(profile.feature_type)}</span>
    </div>
    <div class="mt-12">${renderProfileMetrics(profile)}</div>`;
  const canRebuild = Boolean(modelProfilesState.canRebuildOnSwitch);
  if (rebuildBtn) {
    rebuildBtn.disabled = !canRebuild;
    rebuildBtn.textContent = canRebuild
      ? t('switchRebuild')
      : t('noSessionAudioToRebuild');
  }
  if (status) {
    status.textContent = canRebuild
      ? t('rebuildKeepsSamples')
      : t('clearOnlyValid');
  }
  modal.classList.remove('hidden');
  setTimeout(() => (canRebuild ? rebuildBtn : document.getElementById('modelConfirmClearBtn'))?.focus(), 0);
}

function closeModelSwitchModal() {
  const modal = document.getElementById('modelSwitchModal');
  if (modal) modal.classList.add('hidden');
  pendingModelProfileId = null;
}

function setModelSwitchBusy(busy) {
  ['modelConfirmRebuildBtn', 'modelConfirmClearBtn', 'modelSwitchCloseBtn', 'modelSwitchBtn'].forEach(id => {
    const btn = document.getElementById(id);
    if (!btn) return;
    btn.disabled = busy || (id === 'modelConfirmRebuildBtn' && !modelProfilesState.canRebuildOnSwitch);
  });
  document.querySelectorAll('[data-model-profile-id]').forEach(btn => {
    btn.disabled = busy || btn.classList.contains('missing');
  });
}

function applyThresholdHint(value) {
  const threshold = Number(value);
  if (!Number.isFinite(threshold)) return;
  [
    ['detectThr', 'thrVal'],
    ['longThr', 'longThrVal'],
  ].forEach(([inputId, labelId]) => {
    const input = document.getElementById(inputId);
    const label = document.getElementById(labelId);
    if (!input) return;
    input.value = String(threshold);
    if (label) label.textContent = threshold.toFixed(2);
  });
}

async function confirmModelSwitch(enrollmentPolicy) {
  if (!pendingModelProfileId) return;
  await switchModelProfile(pendingModelProfileId, enrollmentPolicy);
}

async function switchModelProfile(profileId, enrollmentPolicy = 'clear') {
  const profile = getModelProfile(profileId);
  if (!profile) {
    toast('error', t('chooseModelFirst'));
    return;
  }

  setModelSwitchBusy(true);
  const fd = new FormData();
  fd.append('profile_id', profileId);
  fd.append('enrollment_policy', enrollmentPolicy);
  try {
    const r = await fetch(API + '/api/model/select', { method: 'POST', body: fd });
    const d = await r.json();
    if (!r.ok) {
      toast('error', d.error || t('couldNotSwitchModel'));
      return;
    }
    detectionHistory = [];
    renderDetectionHistory();
    applyThresholdHint(d.model?.threshold_hint);
    await refreshEnrolled();
    await loadModelProfiles();
    await loadModelInfo();
    closeModelSwitchModal();
    const rebuilt = d.enrollment?.rebuilt;
    const modelName = d.model?.profile_short_label || d.model?.profile_label || profileId;
    addLog(t(rebuilt ? 'switchedModelLogRebuilt' : 'switchedModelLogCleared', { model: modelName }));
    toast('success', rebuilt ? t('switchedRebuilt') : t('switchedCleared'));
  } catch (err) {
    console.error('switchModelProfile failed', err);
    toast('error', t('switchNetworkError'));
  } finally {
    setModelSwitchBusy(false);
  }
}

function modelInfoLabel(label) {
  return t('modelLabels')?.[label] || label;
}

function renderModelInfo(d) {
  if (!d) return;
  document.getElementById('modelCards').innerHTML = [
    ['Profile', d.profile_label || d.active_profile],
    ['Architecture', d.architecture],
    ['Parameters', d.parameters?.toLocaleString() || '-'],
    ['Embedding', d.embedding_dim],
    ['Feature', d.feature_type],
    ['Device', d.device],
    ['Input', d.input_shape],
    ['Checkpoint', d.checkpoint],
  ].map(([l, v]) => `<div class="metric-card"><div class="metric-value">${escapeHtml(v)}</div><div class="metric-label">${escapeHtml(modelInfoLabel(l))}</div></div>`).join('');
  const sidebarStatus = document.getElementById('sidebarStatus');
  if (sidebarStatus) sidebarStatus.textContent = d.profile_label || d.checkpoint || t('modelLoaded');
  const activeTitle = document.getElementById('activeModelTitle');
  if (activeTitle) activeTitle.textContent = d.profile_short_label || d.profile_label || d.checkpoint || t('modelLoaded');
  const ev = document.getElementById('evalResults');
  if (d.evaluations && Object.keys(d.evaluations).length) {
    ev.innerHTML = Object.entries(d.evaluations).map(([name, data]) => {
      if (typeof data === 'object' && !Array.isArray(data) && data.auc !== undefined) {
        return `<div class="mt-12"><p class="text-sm fw-600 mb-8">${escapeHtml(name)}</p>
          <div class="grid-auto">${[
            ['AUC', data.auc], ['EER', data.eer], ['KW-ACC', data.keyword_acc], ['F1', data.f1]
          ].map(([l, v]) => `<div class="metric-card"><div class="metric-value">${escapeHtml((v || 0).toFixed(3))}</div><div class="metric-label">${escapeHtml(l)}</div></div>`).join('')}</div></div>`;
      }
      return '';
    }).join('') || `<p class="text-muted text-sm">${escapeHtml(t('noEvalData'))}</p>`;
  } else ev.innerHTML = `<p class="text-muted text-sm">${escapeHtml(t('noEvalResults'))}</p>`;
}

async function loadModelInfo() {
  try {
    const r = await fetch(API + '/api/model/info');
    const d = await r.json();
    lastModelInfo = d;
    renderModelInfo(d);
  } catch { document.getElementById('evalResults').innerHTML = `<p class="text-muted text-sm">${escapeHtml(t('couldNotLoad'))}</p>`; }
}

// Presets
async function loadPresets() {
  try {
    const r = await fetch(API + '/api/presets');
    const d = await r.json();
    const wrap = document.getElementById('presetPills');
    wrap.innerHTML = '';
    Object.entries(d.presets).forEach(([name, words]) => {
      const pill = document.createElement('button');
      pill.type = 'button';
      pill.className = 'preset-pill';
      pill.textContent = name;
      pill.addEventListener('click', () => selectPreset(pill, words));
      wrap.appendChild(pill);
    });
  } catch {}
}
function selectPreset(el, words) {
  document.querySelectorAll('.preset-pill').forEach(p => p.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('gscWords').value = words;
  if (words === OPEN_SET_SPLITS.gsc_17_17.known.join(',')) {
    const preset = document.getElementById('osPreset');
    if (preset) preset.value = 'gsc_17_17';
    applyOpenSetPreset();
  }
}

function ratePct(rate) {
  const n = Number(rate);
  return Number.isFinite(n) ? `${(n * 100).toFixed(1)}%` : '-';
}

function compactWordList(words) {
  const list = (words || []).map(String).filter(Boolean);
  if (!list.length) return '';
  return list.length > 14 ? `${list.slice(0, 14).join(', ')}...` : list.join(', ');
}

function compactShortWordList(items) {
  const list = (items || []).map(item => `${item.word} ${item.available}/${item.requested}`);
  return compactWordList(list);
}

function renderOpenSetNotice(label, value, tone = 'warning') {
  if (!value) return '';
  const cls = tone === 'danger' ? 'alert-warning' : '';
  return `<div class="alert ${cls} mb-8"><strong>${escapeHtml(label)}:</strong> ${escapeHtml(value)}</div>`;
}

function renderOpenSetWarnings(data) {
  return [
    renderOpenSetNotice(t('skippedUnknownWords'), compactWordList(data.skipped_unknown_words)),
    renderOpenSetNotice(t('missingKnownWords'), compactWordList(data.missing_known_words), 'danger'),
    renderOpenSetNotice(t('missingUnknownWords'), compactWordList(data.missing_unknown_words), 'danger'),
    renderOpenSetNotice(t('shortAudioWords'), compactShortWordList([...(data.short_known_words || []), ...(data.short_unknown_words || [])])),
  ].join('');
}

function renderOpenSetCaseCard(item) {
  const isCorrect = Boolean(item.correct);
  const statusClass = isCorrect ? 'match-ok' : item.kind === 'unknown' ? 'match-err' : 'match-warn';
  const badgeClass = isCorrect ? 'badge-success' : item.status === 'false_accept' ? 'badge-danger' : 'badge-warn';
  const badgeText = item.status ? item.status.replace(/_/g, ' ').toUpperCase() : (isCorrect ? 'OK' : 'ERR');
  return `<article class="detection-card ${statusClass}">
    <div class="detection-card-head">
      <div>
        <div class="detection-index">${escapeHtml(item.kind || '-')}</div>
        <div class="detection-time">${escapeHtml(item.file || '-')}</div>
      </div>
      <span class="badge ${badgeClass}">${escapeHtml(badgeText)}</span>
    </div>
    <div class="detection-main">
      <div>
        <div class="mini-label">${escapeHtml(t('sourceWord'))}</div>
        <div class="detection-word expected">${escapeHtml(item.word || '-')}</div>
      </div>
      <div>
        <div class="mini-label">${escapeHtml(t('expected'))}</div>
        <div class="detection-word expected">${escapeHtml(item.expected || '-')}</div>
      </div>
      <div>
        <div class="mini-label">${escapeHtml(t('predicted'))}</div>
        <div class="detection-word predicted">${escapeHtml(item.predicted || '-')}</div>
      </div>
    </div>
    <div class="detection-detail-grid">
      <div><span class="mini-label">${escapeHtml(t('l2Dist'))}</span><strong class="font-mono">${escapeHtml(metricText(item.distance, 4))}</strong></div>
      <div><span class="mini-label">${escapeHtml(t('threshold'))}</span><strong class="font-mono">${escapeHtml(metricText(item.threshold, 3))}</strong></div>
      <div><span class="mini-label">${escapeHtml(t('margin'))}</span><strong class="font-mono">${escapeHtml(metricText(item.margin, 4))}</strong></div>
      <div><span class="mini-label">${escapeHtml(t('acceptMargin'))}</span><strong class="font-mono">${escapeHtml(metricText(item.accept_margin, 4))}</strong></div>
      <div><span class="mini-label">${escapeHtml(t('confidence'))}</span><strong class="font-mono">${escapeHtml(metricText(item.confidence, 3))}</strong></div>
      <div><span class="mini-label">${escapeHtml(t('file'))}</span><strong class="font-mono">${escapeHtml(item.file || '-')}</strong></div>
    </div>
    <div class="candidate-box">
      <div class="mini-label">${escapeHtml(t('top3Candidates'))}</div>
      ${renderTopCandidates(item.top_3 || [])}
    </div>
  </article>`;
}

function renderOpenSetCaseSection(title, items, emptyText) {
  const rows = items || [];
  return `<div class="mt-16">
    <div class="card-header compact-header">
      <div class="card-title">${escapeHtml(title)}</div>
      <span class="badge badge-neutral">${rows.length}</span>
    </div>
    ${rows.length
      ? `<div class="detection-card-grid">${rows.map(renderOpenSetCaseCard).join('')}</div>`
      : `<div class="empty-state compact"><p>${escapeHtml(emptyText)}</p></div>`}
  </div>`;
}

function wordsFromInput(id) {
  const el = document.getElementById(id);
  if (!el) return [];
  return el.value.split(/[\s,]+/).map(w => w.trim()).filter(Boolean);
}

function setRangeValue(id, value, digits = 2) {
  const input = document.getElementById(id);
  const label = document.getElementById(`${id}Val`);
  if (!input) return;
  input.value = String(value);
  if (label) label.textContent = Number(value).toFixed(digits);
}

function updateOpenSetSplitPreview() {
  const preview = document.getElementById('osSplitPreview');
  if (!preview) return;
  const known = wordsFromInput('osKnownWords');
  const unknown = wordsFromInput('osWords');
  const preset = document.getElementById('osPreset')?.value || 'manual';
  const heldout = OPEN_SET_SPLITS[preset]?.heldout?.join(', ') || '-';
  preview.textContent = t('openSetSplitSummary', {
    known: known.length,
    unknown: unknown.length,
    heldout,
  });
}

function applyOpenSetPreset() {
  const preset = document.getElementById('osPreset')?.value || 'manual';
  const split = OPEN_SET_SPLITS[preset];
  if (split) {
    const known = split.known.join(',');
    const unknown = split.unknown.join(',');
    document.getElementById('osKnownWords').value = known;
    document.getElementById('osWords').value = unknown;
    const gscWords = document.getElementById('gscWords');
    if (gscWords) gscWords.value = known;
  }
  updateOpenSetSplitPreview();
}

function renderWordSetCard(label, words) {
  const list = words || [];
  return `<div class="alert mb-8">
    <strong>${escapeHtml(label)}:</strong>
    <span>${escapeHtml(compactWordList(list)) || '-'}</span>
  </div>`;
}

function openSetFormData() {
  const fd = new FormData();
  const closeGuard = document.getElementById('osCloseWordGuard')?.checked;
  fd.append('preset', document.getElementById('osPreset')?.value || 'manual');
  fd.append('known_words', document.getElementById('osKnownWords')?.value || '');
  fd.append('unknown_words', document.getElementById('osWords')?.value || '');
  fd.append('samples_per_word', document.getElementById('osK').value);
  fd.append('threshold', document.getElementById('osThr').value);
  fd.append('accept_margin', closeGuard ? document.getElementById('osAcceptMargin').value : '0');
  fd.append('use_per_class', checkboxFormValue('osPerClass'));
  fd.append('use_close_word_guard', checkboxFormValue('osCloseWordGuard'));
  fd.append('seed', document.getElementById('osSeed').value || '1234');
  return fd;
}

function renderOpenSetSummary(data) {
  const s = data.summary || {};
  const settings = data.settings || {};
  const guardOn = settings.close_word_guard !== false;
  const perClassOn = settings.use_per_class !== false;
  return `<div class="result-summary-grid mb-16">
    ${renderResultMetric(t('knownTested'), String(s.known_tested ?? 0))}
    ${renderResultMetric(t('unknownTested'), String(s.unknown_tested ?? 0))}
    ${renderResultMetric(t('candidateLabels'), String((data.candidate_words || []).length))}
    ${renderResultMetric(t('balancedScore'), ratePct(s.balanced_score), Number(s.balanced_score) >= 0.8 ? 'good' : 'warn')}
    ${renderResultMetric(t('openSetAcc'), ratePct(s.open_set_acc), Number(s.open_set_acc) >= 0.8 ? 'good' : 'warn')}
    ${renderResultMetric(t('keywordAcc'), ratePct(s.keyword_acc), Number(s.keyword_acc) >= 0.8 ? 'good' : 'warn')}
    ${renderResultMetric(t('unknownRejectAcc'), ratePct(s.unknown_reject_acc), Number(s.unknown_reject_acc) >= 0.8 ? 'good' : 'warn')}
    ${renderResultMetric(t('falseAcceptRate'), ratePct(s.false_accept_rate), Number(s.false_accept_rate) > 0 ? 'warn' : 'good')}
    ${renderResultMetric(t('falseRejectRate'), ratePct(s.false_reject_rate), Number(s.false_reject_rate) > 0 ? 'warn' : 'good')}
    ${renderResultMetric(t('knownMisses'), String(s.known_misses ?? 0), Number(s.known_misses) ? 'warn' : 'good')}
    ${renderResultMetric(t('perClassThreshold'), perClassOn ? t('perClassOn') : t('perClassOff'), perClassOn ? '' : 'warn')}
    ${renderResultMetric(t('closeWordGuard'), guardOn ? t('closeWordGuardOn') : t('closeWordGuardOff'), guardOn ? '' : 'warn')}
    ${renderResultMetric(t('acceptMargin'), metricText(settings.accept_margin, 4), Number(settings.accept_margin) === 0 ? 'warn' : '')}
  </div>`;
}

function renderOpenSetSplit(data) {
  return [
    renderWordSetCard(t('knownWords'), data.known_words || []),
    renderWordSetCard(t('unknownWords'), data.unknown_words || []),
    renderWordSetCard('Holdout', data.heldout_words || []),
  ].join('');
}

function calibrationSettingsId(kind) {
  return `os-cal-${kind}`;
}

function renderCalibrationOption(title, row, id) {
  if (!row) return '';
  openSetCalibrationCache[id] = row;
  return `<div class="metric-card calibration-option">
    <div class="metric-value">${escapeHtml(ratePct(row.balanced_score))}</div>
    <div class="metric-label">${escapeHtml(title)}</div>
    <div class="text-xs text-muted mt-8">
      ${escapeHtml(t('keywordAcc'))}: ${escapeHtml(ratePct(row.keyword_acc))}<br>
      ${escapeHtml(t('unknownRejectAcc'))}: ${escapeHtml(ratePct(row.unknown_reject_acc))}<br>
      ${escapeHtml(t('thresholdValue'))}: ${escapeHtml(metricText(row.threshold, 3))}<br>
      ${escapeHtml(t('acceptMargin'))}: ${escapeHtml(metricText(row.accept_margin, 4))}<br>
      ${escapeHtml(t('perClass'))}: ${escapeHtml(row.use_per_class ? 'ON' : 'OFF')}
    </div>
    <button class="btn btn-ghost btn-sm mt-12" type="button" onclick="applyOpenSetCalibrationSettings('${escapeHtml(id)}')">${escapeHtml(t('applySettings'))}</button>
  </div>`;
}

function applyOpenSetCalibrationSettings(id) {
  const row = openSetCalibrationCache[id];
  if (!row) return;
  setRangeValue('osThr', row.threshold, 2);
  setRangeValue('osAcceptMargin', row.accept_margin, 2);
  document.getElementById('osPerClass').checked = Boolean(row.use_per_class);
  document.getElementById('osCloseWordGuard').checked = Number(row.accept_margin) > 0;
  toast('success', t('settingsApplied'));
}

async function runOpenSetTest() {
  const out = document.getElementById('osResult');
  const fd = openSetFormData();

  setBusy('osRunBtn', true, t('runningOpenSet'));
  out.innerHTML = '<div class="flex-center" style="padding:40px"><div class="spinner"></div></div>';
  try {
    const r = await fetch(API + '/api/open-set/test', { method: 'POST', body: fd });
    const data = await r.json();
    if (!r.ok) {
      out.innerHTML = `<div class="card">
        <div class="alert alert-warning mb-16">${escapeHtml(data.error || t('failed'))}</div>
        ${renderOpenSetWarnings(data)}
      </div>`;
      toast('error', data.error || t('failed'));
      return;
    }

    const s = data.summary || {};
    const settings = data.settings || {};
    const html = `<div class="card">
      <div class="card-header">
        <div class="card-title">${escapeHtml(t('openSetTitle'))}</div>
        <span class="badge badge-neutral">${escapeHtml(settings.engine || 'open_set')}</span>
      </div>
      ${renderOpenSetSummary(data)}
      ${renderOpenSetSplit(data)}
      ${renderOpenSetWarnings(data)}
      ${renderOpenSetCaseSection(t('falseAccepts'), data.false_accepts || [], t('noFalseAccepts'))}
      ${renderOpenSetCaseSection(t('knownMisses'), data.known_misses || [], t('noKnownMisses'))}
    </div>`;
    out.innerHTML = html;
    toast('success', t('openSetEvaluated', { accuracy: ratePct(s.open_set_acc) }));
  } catch {
    out.innerHTML = '';
    toast('error', t('networkError'));
  } finally {
    setBusy('osRunBtn', false);
  }
}

async function runOpenSetCalibration() {
  const out = document.getElementById('osResult');
  const fd = openSetFormData();
  fd.append('threshold_min', '0.10');
  fd.append('threshold_max', '1.20');
  fd.append('threshold_step', '0.05');
  fd.append('accept_margin_values', '0.00,0.02,0.05,0.08,0.10');
  fd.append('use_per_class_options', 'true,false');

  setBusy('osCalibrateBtn', true, t('runningCalibration'));
  out.innerHTML = '<div class="flex-center" style="padding:40px"><div class="spinner"></div></div>';
  try {
    const r = await fetch(API + '/api/open-set/calibrate', { method: 'POST', body: fd });
    const data = await r.json();
    if (!r.ok) {
      out.innerHTML = `<div class="card">
        <div class="alert alert-warning mb-16">${escapeHtml(data.error || t('failed'))}</div>
        ${renderOpenSetWarnings(data)}
      </div>`;
      toast('error', data.error || t('failed'));
      return;
    }
    openSetCalibrationCache = {};
    const bestBalancedId = calibrationSettingsId('balanced');
    const bestOpenId = calibrationSettingsId('open');
    const bestKeywordId = calibrationSettingsId('keyword');
    out.innerHTML = `<div class="card">
      <div class="card-header">
        <div class="card-title">${escapeHtml(t('calibrationResults'))}</div>
        <span class="badge badge-neutral">${escapeHtml(data.settings?.engine || 'calibration')}</span>
      </div>
      ${renderOpenSetSplit(data)}
      ${renderOpenSetWarnings(data)}
      <div class="grid-auto">
        ${renderCalibrationOption(t('bestBalanced'), data.best_balanced, bestBalancedId)}
        ${renderCalibrationOption(t('bestRejectUnknown'), data.best_open_set, bestOpenId)}
        ${renderCalibrationOption(t('bestRecognizeKeyword'), data.best_keyword, bestKeywordId)}
      </div>
      <div class="mt-16">
        <div class="card-header compact-header">
          <div class="card-title">${escapeHtml(t('calibrationResults'))}</div>
          <span class="badge badge-neutral">${escapeHtml(String((data.rows || []).length))}</span>
        </div>
        <div class="table-scroll">
          <table class="data-table compact-table">
            <thead><tr>
              <th>${escapeHtml(t('thresholdValue'))}</th>
              <th>${escapeHtml(t('acceptMargin'))}</th>
              <th>${escapeHtml(t('perClass'))}</th>
              <th>${escapeHtml(t('balancedScore'))}</th>
              <th>${escapeHtml(t('keywordAcc'))}</th>
              <th>${escapeHtml(t('unknownRejectAcc'))}</th>
              <th>${escapeHtml(t('falseAcceptRate'))}</th>
              <th>${escapeHtml(t('falseRejectRate'))}</th>
            </tr></thead>
            <tbody>${(data.rows || []).slice(0, 40).map(row => `<tr>
              <td>${escapeHtml(metricText(row.threshold, 3))}</td>
              <td>${escapeHtml(metricText(row.accept_margin, 4))}</td>
              <td>${escapeHtml(row.use_per_class ? 'ON' : 'OFF')}</td>
              <td>${escapeHtml(ratePct(row.balanced_score))}</td>
              <td>${escapeHtml(ratePct(row.keyword_acc))}</td>
              <td>${escapeHtml(ratePct(row.unknown_reject_acc))}</td>
              <td>${escapeHtml(ratePct(row.false_accept_rate))}</td>
              <td>${escapeHtml(ratePct(row.false_reject_rate))}</td>
            </tr>`).join('')}</tbody>
          </table>
        </div>
      </div>
    </div>`;
    toast('success', t('openSetCalibrated', { score: ratePct(data.best_balanced?.balanced_score) }));
  } catch (err) {
    console.error('runOpenSetCalibration failed', err);
    out.innerHTML = '';
    toast('error', t('networkError'));
  } finally {
    setBusy('osCalibrateBtn', false);
  }
}

// Batch evaluation
let batchTxtFile = null;

async function runBatchEval() {
  if (!batchTxtFile) { toast('error', t('uploadGroundTruthFirst')); return; }
  const out = document.getElementById('batchResult');
  const summary = document.getElementById('batchSummary');
  out.innerHTML = '<div class="flex-center" style="padding:40px"><div class="spinner"></div></div>';
  summary.classList.add('hidden');

  const fd = new FormData();
  fd.append('labels_file', batchTxtFile, batchTxtFile.name);
  fd.append('threshold', document.getElementById('batchThr').value);
  fd.append('use_per_class', document.getElementById('batchPerClass').checked);

  try {
    const r = await fetch(API + '/api/detect/batch', { method: 'POST', body: fd });
    const d = await r.json();
    if (!r.ok) { toast('error', d.error || t('failed')); out.innerHTML = ''; return; }

    // Summary
    document.getElementById('batchTotal').textContent = d.total;
    document.getElementById('batchCorrect').textContent = d.correct;
    document.getElementById('batchAccuracy').textContent = d.accuracy + '%';
    summary.classList.remove('hidden');

    // Table
    let html = `<div class="card"><div class="card-header"><div class="card-title">${escapeHtml(t('results'))}</div>
      <span class="badge ${d.accuracy >= 80 ? 'badge-success' : d.accuracy >= 50 ? 'badge-warn' : 'badge-danger'}">${d.accuracy}% ${escapeHtml(t('accuracy').toLowerCase())}</span></div>
      <table class="data-table"><thead><tr>
        <th>#</th><th>File</th><th>${escapeHtml(t('expected'))}</th><th>${escapeHtml(t('predicted'))}</th><th>${escapeHtml(t('distanceMetric'))}</th><th>${escapeHtml(t('status'))}</th>
      </tr></thead><tbody>`;
    d.results.forEach((r, i) => {
      const icon = r.correct ? 'OK' : 'ERR';
      const cls = r.correct ? 'badge-success' : r.status === 'file_not_found' ? 'badge-neutral' : 'badge-danger';
      const label = r.correct ? t('correct') : r.status === 'file_not_found' ? t('notFound') : t('wrong');
      html += `<tr>
        <td>${i + 1}</td>
        <td class="font-mono text-xs">${escapeHtml(r.file)}</td>
        <td><strong>${escapeHtml(r.expected)}</strong></td>
        <td>${escapeHtml(r.predicted)}</td>
        <td class="font-mono">${escapeHtml(r.distance || '-')}</td>
        <td><span class="badge ${cls}">${icon} ${label}</span></td>
      </tr>`;
    });
    html += '</tbody></table></div>';
    out.innerHTML = html;
    toast('success', t('evaluatedFiles', { total: d.total, accuracy: d.accuracy }));
  } catch { toast('error', t('networkError')); out.innerHTML = ''; }
}

// Init
document.addEventListener('DOMContentLoaded', () => {
  applyLanguage(currentLang);
  setupDrop('detectDrop', 'detectFile', 'detect');
  setupDrop('longDrop', 'longFile', 'long');
  const modelModal = document.getElementById('modelSwitchModal');
  if (modelModal) {
    modelModal.addEventListener('click', event => {
      if (event.target === modelModal) closeModelSwitchModal();
    });
  }
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape') closeModelSwitchModal();
  });
  document.getElementById('micFile').addEventListener('change', function () {
    if (this.files.length) {
      audioBlobs.enroll = this.files[0];
      document.getElementById('micStatus').textContent = t('loadedFile', { name: this.files[0].name });
      autoEnrollMic();
    }
  });

  // Batch file upload
  const batchInput = document.getElementById('batchFile');
  const batchDrop = document.getElementById('batchDrop');
  if (batchInput && batchDrop) {
    batchInput.addEventListener('change', () => {
      if (batchInput.files.length) {
        batchTxtFile = batchInput.files[0];
        batchDrop.querySelector('p').textContent = t('loadedFile', { name: batchTxtFile.name });
        toast('success', t('fileLoaded'));
      }
    });
    batchDrop.addEventListener('dragover', e => { e.preventDefault(); batchDrop.classList.add('dragover'); });
    batchDrop.addEventListener('dragleave', () => batchDrop.classList.remove('dragover'));
    batchDrop.addEventListener('drop', e => {
      e.preventDefault(); batchDrop.classList.remove('dragover');
      if (e.dataTransfer.files.length) {
        batchTxtFile = e.dataTransfer.files[0];
        batchDrop.querySelector('p').textContent = t('loadedFile', { name: batchTxtFile.name });
        toast('success', t('fileLoaded'));
      }
    });
  }

  // Ground truth file upload (Long File tab)
  const gtInput = document.getElementById('gtFile');
  const gtDrop = document.getElementById('gtDrop');
  if (gtInput && gtDrop) {
    gtInput.addEventListener('change', async () => {
      if (gtInput.files.length) {
        try {
          await handleGroundTruthFile(gtInput.files[0]);
        } catch (err) {
          console.error('ground truth file load failed', err);
          toast('error', t('failed'));
        }
      }
    });
    gtDrop.addEventListener('dragover', e => { e.preventDefault(); gtDrop.classList.add('dragover'); });
    gtDrop.addEventListener('dragleave', () => gtDrop.classList.remove('dragover'));
    gtDrop.addEventListener('drop', async e => {
      e.preventDefault(); gtDrop.classList.remove('dragover');
      if (e.dataTransfer.files.length) {
        try {
          await handleGroundTruthFile(e.dataTransfer.files[0]);
        } catch (err) {
          console.error('ground truth file drop failed', err);
          toast('error', t('failed'));
        }
      }
    });
  }

  const timingInput = document.getElementById('timingFile');
  const timingDrop = document.getElementById('timingDrop');
  if (timingInput && timingDrop) {
    timingInput.addEventListener('change', async () => {
      if (timingInput.files.length) {
        try {
          await handleTimingFile(timingInput.files[0]);
        } catch (err) {
          console.error('timing file load failed', err);
          toast('error', t('invalidTimingJson'));
        }
      }
    });
    timingDrop.addEventListener('dragover', e => { e.preventDefault(); timingDrop.classList.add('dragover'); });
    timingDrop.addEventListener('dragleave', () => timingDrop.classList.remove('dragover'));
    timingDrop.addEventListener('drop', async e => {
      e.preventDefault(); timingDrop.classList.remove('dragover');
      if (e.dataTransfer.files.length) {
        try {
          await handleTimingFile(e.dataTransfer.files[0]);
        } catch (err) {
          console.error('timing file drop failed', err);
          toast('error', t('invalidTimingJson'));
        }
      }
    });
  }

  ['osKnownWords', 'osWords'].forEach(id => {
    const input = document.getElementById(id);
    if (input) input.addEventListener('input', updateOpenSetSplitPreview);
  });
  applyOpenSetPreset();

  loadPresets();
  loadProfileList();
  refreshEnrolled();
  loadModelProfiles();
  loadModelInfo();
});

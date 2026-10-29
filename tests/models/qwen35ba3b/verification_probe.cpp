// Focused diagnostic, not a substitute for qwen35ba3b_target_test.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/tokenizer.hpp"

namespace {
using Executor = gufo::hip::QwenGpuExecutor;
using Token = gufo::tokenization::TokenId;
constexpr std::array<const char*, 2> kTexts{
    "Virtual memory lets an operating system give each process its own address "
    "space. Pages map virtual addresses to physical memory. A page fault "
    "occurs when the requested page is unavailable.",
    "A train leaves the station at noon and travels at sixty kilometers per "
    "hour. Another train leaves one hour later at ninety kilometers per hour.",
};

std::size_t Compare(std::string label, std::span<const float> expected,
                    std::span<const float> actual) {
  if (expected.size() != actual.size())
    throw std::runtime_error("diagnostic size mismatch: " + label);
  std::size_t different = 0;
  float max_abs = 0.0F;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (std::memcmp(&expected[i], &actual[i], sizeof(float)) != 0) {
      ++different;
      max_abs = std::max(max_abs, std::fabs(expected[i] - actual[i]));
    }
  }
  std::cout << label << " different=" << different << " max_abs=" << max_abs
            << '\n';
  return different;
}

struct Session {
  std::unique_ptr<Executor> executor;
  std::vector<Token> tokens;
  std::uint32_t position;
  std::unique_ptr<gufo::hip::QwenGpuSnapshot> snapshot;
  std::vector<std::vector<float>> logits;
  std::vector<float> hidden;
};
}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr
        << "usage: qwen35ba3b_verification_probe MODEL serial|concurrent\n";
    return 2;
  }
  try {
    const bool concurrent = std::string(argv[2]) == "concurrent";
    if (!concurrent && std::string(argv[2]) != "serial")
      throw std::runtime_error("unknown diagnostic mode");
    std::string error;
    const std::shared_ptr<const gufo::core::GgufReader> reader =
        gufo::core::GgufReader::OpenFile(argv[1], &error);
    if (!reader)
      throw std::runtime_error(error);
    const auto model = gufo::hip::QwenGpuModel::CreateFromGguf(reader, &error);
    if (!model)
      throw std::runtime_error(error);
    const auto& config = model->GetConfig();
    const std::size_t taps = std::min<std::size_t>(16, config.num_layers);
    std::vector<std::uint32_t> layers;
    for (std::uint32_t i = 0; i < taps; ++i)
      layers.push_back(config.num_layers - static_cast<std::uint32_t>(taps) +
                       i);
    const std::size_t width = taps * config.hidden_size;
    const std::array<std::size_t, 3> rows{8, 8, 7};
    std::vector<Session> sessions;
    const std::size_t count = concurrent ? 3 : 1;
    std::size_t differing = 0;
    for (std::size_t s = 0; s < count; ++s) {
      auto& session = sessions.emplace_back();
      session.executor = Executor::Create(model, &error, 4096);
      if (!session.executor)
        throw std::runtime_error(error);
      auto& executor = *session.executor;
      executor.SetPromptHiddenCapture(true, layers);
      session.tokens =
          executor.GetTokenizer().Encode(kTexts[s % kTexts.size()]);
      session.position =
          static_cast<std::uint32_t>(session.tokens.size() - rows[s]);
      executor.Reset();
      (void)executor.ForwardPromptBatch(
          std::span(session.tokens).first(session.position));
      session.snapshot = executor.SaveSnapshot(session.position);
      const auto candidates =
          std::span(session.tokens).subspan(session.position);
      if (concurrent) {
        (void)executor.ForwardVerificationChunk(candidates, session.position,
                                                true);
        const auto hidden = executor.GetVerificationHiddenStates();
        session.hidden.assign(hidden.begin(), hidden.end());
        for (std::size_t r = 0; r < rows[s]; ++r) {
          const auto logits = executor.CopyVerificationLogits(r);
          session.logits.emplace_back(logits.begin(), logits.end());
        }
      } else {
        for (std::size_t r = 0; r < rows[s]; ++r) {
          (void)executor.ForwardToken(
              candidates[r], session.position + static_cast<std::uint32_t>(r));
          const auto logits = executor.CopyLastLogits();
          session.logits.emplace_back(logits.begin(), logits.end());
          const auto hidden = executor.CopyLastHidden();
          session.hidden.insert(session.hidden.end(), hidden.begin(),
                                hidden.end());
        }
        executor.RestoreSnapshot(*session.snapshot);
        for (std::size_t r = 0; r < rows[s]; ++r) {
          (void)executor.ForwardToken(
              candidates[r], session.position + static_cast<std::uint32_t>(r));
          differing += Compare("snapshot-replay row=" + std::to_string(r),
                               session.logits[r], executor.CopyLastLogits());
          if (r == 0) {
            const auto actual = executor.CopyLastHidden();
            for (std::size_t layer = 0; layer < taps; ++layer) {
              const auto offset = layer * config.hidden_size;
              Compare(
                  "snapshot-replay row=0 layer=" +
                      std::to_string(layers[layer]),
                  std::span(session.hidden).subspan(offset, config.hidden_size),
                  actual.subspan(offset, config.hidden_size));
            }
          }
        }
      }
      executor.RestoreSnapshot(*session.snapshot);
    }
    std::vector<gufo::hip::QwenGpuVerificationItem> items;
    for (auto& session : sessions)
      items.push_back({session.executor.get(),
                       std::span(session.tokens).subspan(session.position),
                       session.position, true});
    (void)Executor::ForwardVerificationBatch(items);
    for (std::size_t s = 0; s < count; ++s) {
      const auto actual = sessions[s].executor->GetVerificationHiddenStates();
      if (actual.size() != rows[s] * width)
        throw std::runtime_error("hidden capture size mismatch");
      for (std::size_t layer = 0; layer < taps; ++layer) {
        const auto offset = layer * config.hidden_size;
        Compare(
            "session=" + std::to_string(s) +
                " row=0 layer=" + std::to_string(layers[layer]),
            std::span(sessions[s].hidden).subspan(offset, config.hidden_size),
            actual.subspan(offset, config.hidden_size));
      }
      for (std::size_t r = 0; r < rows[s]; ++r)
        differing += Compare("verification session=" + std::to_string(s) +
                                 " row=" + std::to_string(r),
                             sessions[s].logits[r],
                             sessions[s].executor->CopyVerificationLogits(r));
    }
    return differing == 0 ? 0 : 1;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

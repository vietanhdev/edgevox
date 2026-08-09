import { defineConfig } from "vitepress";
import { withMermaid } from "vitepress-plugin-mermaid";

// `withMermaid` takes ONE config object carrying a `mermaid` key. Passing the
// mermaid options as a second argument silently drops them — which is why every
// diagram rendered in mermaid's stock lavender until 2026-08-09.
export default withMermaid({
  ...defineConfig({
    title: "EdgeVox",
    description: "Offline voice agent framework for robots — agents, skills, workflows, 2D/3D simulation, sub-second voice pipeline",
    lang: "en-US",

    // Docs live directly in this directory
    srcDir: ".",
    cleanUrls: true,

    // ``reports/`` holds raw benchmark data (JSON) we don't want on the
    // public site — excluded from the build. Refined report pages live
    // under ``documentation/reports/``.
    srcExclude: ["reports/**"],

    // The raw data directory is the only tolerated dead-link target.
    // Also skip ``http://localhost…`` example URLs in the monitoring
    // guide — those are instructions, not real links.
    ignoreDeadLinks: ["localhost", /^\/reports\//, /^https?:\/\/(localhost|127\.0\.0\.1)/],

    head: [
      ["link", { rel: "icon", type: "image/svg+xml", href: "/logo.svg" }],
      [
        "meta",
        { name: "theme-color", content: "#a54c00" },
      ],
    ],

    appearance: false,

    themeConfig: {
      logo: "/logo.svg",
      siteTitle: "EdgeVox",

      search: {
        provider: "local",
      },

      nav: [
        { text: "Documentation", link: "/documentation/" },
        {
          text: "Links",
          items: [
            { text: "GitHub", link: "https://github.com/nrl-ai/edgevox" },
            { text: "PyPI", link: "https://pypi.org/project/edgevox" },
          ],
        },
      ],

      sidebar: {
        "/documentation/": [
          {
            text: "Start Here",
            items: [
              { text: "Introduction", link: "/documentation/" },
              { text: "Quick Start", link: "/documentation/quickstart" },
              { text: "Configuration", link: "/documentation/configuration" },
            ],
          },
          {
            text: "Agents",
            items: [
              { text: "Agents & Tools", link: "/documentation/agents" },
              { text: "Agent loop", link: "/documentation/agent-loop" },
              { text: "Hooks", link: "/documentation/hooks" },
              { text: "Memory", link: "/documentation/memory" },
              { text: "Multi-agent", link: "/documentation/multiagent" },
              { text: "Tool calling", link: "/documentation/tool-calling" },
              { text: "Workflow recipes", link: "/documentation/workflow-recipes" },
              { text: "Interrupt & barge-in", link: "/documentation/interrupt" },
            ],
          },
          {
            text: "Voice & Audio",
            items: [
              { text: "Voice Pipeline", link: "/documentation/pipeline" },
              { text: "Languages", link: "/documentation/languages" },
              { text: "TUI Commands", link: "/documentation/commands" },
            ],
          },
          {
            text: "Applications",
            items: [
              { text: "RookApp (Desktop)", link: "/documentation/desktop" },
              { text: "Robotics Examples", link: "/documentation/robotics" },
              { text: "ROS2 Integration", link: "/documentation/ros2" },
            ],
          },
          {
            text: "Architecture",
            collapsed: true,
            items: [
              { text: "System Architecture", link: "/documentation/architecture" },
              { text: "Component Design", link: "/documentation/components" },
            ],
          },
          {
            text: "Operations",
            items: [
              { text: "Monitoring & Logging", link: "/documentation/monitoring" },
              {
                text: "SLM tool-calling benchmark",
                link: "/documentation/reports/slm-tool-calling-benchmark",
              },
              {
                text: "Chess commentary benchmark",
                link: "/documentation/reports/chess-commentary-benchmark",
              },
              {
                text: "Robot tool-calling benchmark",
                link: "/documentation/reports/robot-tool-calling-benchmark",
              },
            ],
          },
        ],
      },

      socialLinks: [
        { icon: "github", link: "https://github.com/nrl-ai/edgevox" },
      ],

      footer: {
        message: "Offline voice agent framework for robots",
        copyright: "MIT License",
      },

      editLink: {
        pattern: "https://github.com/nrl-ai/edgevox/edit/main/docs/:path",
        text: "Edit this page on GitHub",
      },
    },
  }),
  // "Ink & Signal" diagram styling — neutral ink-on-paper nodes with burnt
  // orange reserved for the one layer that should pull the eye. Ported from
  // www.vietanh.dev so diagrams read the same across every property.
  mermaid: {
    theme: "base",
    themeVariables: {
      primaryColor: "#f1f1ef", // warm-100 node fill
      primaryTextColor: "#1a1a1d", // ink text
      primaryBorderColor: "#1a1a1d", // ink border (crisp)
      lineColor: "#525250", // neutral ink-gray edges
      secondaryColor: "#f3bd92", // signal-200 accent node
      secondaryBorderColor: "#8c4000", // signal-600
      secondaryTextColor: "#1a1a1d",
      tertiaryColor: "#e3e3e1", // warm-200
      tertiaryBorderColor: "#a6a6a3",
      background: "#fafaf8", // paper
      mainBkg: "#f1f1ef",
      nodeBorder: "#1a1a1d",
      nodeTextColor: "#1a1a1d",
      clusterBkg: "#fafaf8", // paper inside subgraphs, so nodes read as a layer above
      clusterBorder: "#cececb",
      titleColor: "#1a1a1d",
      edgeLabelBackground: "#fafaf8",
      // System stack, NOT the Inter webfont: mermaid measures label widths at
      // render time, so a font that swaps in later renders wider than the box
      // it was measured for and the label is clipped mid-word.
      fontFamily: "ui-sans-serif, system-ui, sans-serif",
      fontSize: "14px",
    },
    // Site chrome uses soft corners; mermaid draws sharp rects by default, so
    // round them here. rx/ry are SVG2 geometry properties, settable from CSS.
    themeCSS: `
      .node rect, .node .label-container { rx: 6px; ry: 6px; }
      .cluster rect { rx: 8px; ry: 8px; }
    `,
    // wrappingWidth is the important one: mermaid wraps node labels at 200px by
    // default, which breaks an ordinary sentence across three lines and
    // hyphenates mid-word. Most diagrams here are top-down, where horizontal
    // room is cheap and vertical room is the scarce resource.
    flowchart: {
      useMaxWidth: false,
      htmlLabels: true,
      wrappingWidth: 340,
      nodeSpacing: 56,
      rankSpacing: 48,
      padding: 14,
      diagramPadding: 12,
      curve: "basis",
    },
    sequence: { useMaxWidth: false, diagramMarginX: 12, diagramMarginY: 12 },
    er: { useMaxWidth: false },
  },
});
